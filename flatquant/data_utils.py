import os
import pickle
import random

import torch
from tqdm import tqdm

from torch.utils.data import DataLoader, TensorDataset

import datasets

from flatquant import utils


class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids


def get_wikitext2(nsamples, seed, seqlen, tokenizer, eval_mode=False):
    if eval_mode:
        testdata = datasets.load_dataset(
            "Salesforce/wikitext", "wikitext-2-raw-v1", split="test"
        )
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        return testenc
    else:
        traindata = datasets.load_dataset(
            "Salesforce/wikitext", "wikitext-2-raw-v1", split="train"
        )
        traindata = traindata.filter(lambda x: len(x) > 0)
        traindata = traindata.map(lambda x: {"text": x["text"].strip()})
        trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
        # random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


def get_c4_new(nsamples, seed, seqlen, tokenizer, eval_mode=False):
    if eval_mode:
        valdata = datasets.load_dataset(
            "./datasets/allenai/c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
        )
        valenc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")
        valenc = valenc.input_ids[:, : (256 * seqlen)]
        valenc = TokenizerWrapper(valenc)
        return valenc
    else:
        traindata = datasets.load_dataset(
            "./datasets/allenai/c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
        )
        # random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
                if trainenc.input_ids.shape[1] >= seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


def get_ptb_new(nsamples, seed, seqlen, tokenizer, eval_mode=False):
    if eval_mode:
        testdata = datasets.load_dataset(
            "./datasets/ptb_text_only", "penn_treebank", split="test"
        )
        testenc = tokenizer(" ".join(testdata["sentence"]), return_tensors="pt")
        return testenc
    else:
        traindata = datasets.load_dataset(
            "./datasets/ptb_text_only", "penn_treebank", split="train"
        )
        trainenc = tokenizer(" ".join(traindata["sentence"]), return_tensors="pt")
        # random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


def get_pile(nsamples, seed, seqlen, tokenizer):
    traindata = datasets.load_dataset("./datasets/pile-val-backup", split="validation")
    trainenc = tokenizer("\n\n".join(traindata["text"][:1000]), return_tensors="pt")
    # random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader


def get_coco(
    nsamples: int, max_seq_len: int, tokenizer, text_encoder, batch_size: int = 32
) -> DataLoader:
    """
    Returns a DataLoader of embedded COCO prompts, processing them in batches to manage memory

    Args:
        nsamples: # samples to return
        max_seq_len: max seq length of samples to return
        tokenizer: tokenizer to use
        text_encoder: text encoder to use
        batch_size: size of batches for processing prompts

    Returns:
        DataLoader containing text embeddings and their lengths
    """

    text_encoder.to(utils.DEV)

    # Read all prompts
    with open("./datasets/coco.txt") as f:
        all_prompts = f.readlines()

    valid_outputs = []
    valid_lens = []
    total_processed = 0

    # Process prompts in batches
    for batch_start in tqdm(
        range(0, len(all_prompts), batch_size), desc="Embedding captions..."
    ):
        batch_end = min(batch_start + batch_size, len(all_prompts))
        batch_prompts = all_prompts[batch_start:batch_end]

        # Tokenize batch with explicit max_length
        tokens = tokenizer.batch_encode_plus(
            batch_prompts,
            return_tensors="pt",
            padding="max_length",
            max_length=max_seq_len,
            truncation=True,
        ).to(text_encoder.device)

        # Get sequence lengths for this batch
        seq_lens = tokens["attention_mask"].sum(dim=1)  # (batch_size,)

        # Process through encoder
        with torch.no_grad():
            embeddings = text_encoder(**tokens).last_hidden_state

        # Filter by sequence length
        batch_valid_mask = seq_lens <= max_seq_len
        batch_valid_outputs = embeddings[batch_valid_mask]
        batch_valid_lens = seq_lens[batch_valid_mask]

        valid_outputs.append(batch_valid_outputs.cpu())
        valid_lens.append(batch_valid_lens.cpu())

        total_processed += len(batch_valid_outputs)
        if total_processed >= nsamples:
            break

    # Concatenate all batches
    all_outputs = torch.cat(valid_outputs, dim=0)[:nsamples]
    all_lens = torch.cat(valid_lens, dim=0)[:nsamples]

    dataset = TensorDataset(all_outputs, all_lens)
    return DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)


def get_loaders(
    args,
    name,
    tokenizer,
    text_encoder,
    nsamples=128,
    seed=0,
    seqlen=2048,
    eval_mode=False,
    batch_size=16,
):
    cache_dir = os.path.join(args.cache_dir, name)
    os.makedirs(cache_dir, exist_ok=True)
    cached_dataset = os.path.join(
        cache_dir, "testset.pkl" if eval_mode else f"trainset-{nsamples}-{seed}.pkl"
    )
    if os.path.exists(cached_dataset):
        print(f"Loading cached tokenized dataset at {cached_dataset}...")
        with open(cached_dataset, "rb") as f:
            dataset = pickle.load(f)
    else:
        if "coco" in name:
            dataset = get_coco(
                nsamples, seqlen, tokenizer, text_encoder, batch_size=batch_size
            )

        with open(cached_dataset, "wb") as f:
            print(f"Saving cached tokenized dataset at {cached_dataset}...")
            pickle.dump(dataset, f)

    return dataset

import os
import pickle
import random

import transformers

import datasets


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


def get_coco(nsamples, seq_len, tokenizer):
    """
    Returns a list of COCO prompts, tokenized by the tokenizer

    This format is kind of dumb but also there are like not that many prompts

    args:
        nsamples: # samples to return
        seq_len: max seq length of samples to return
        tokenizer: tokenizer to use

    returns:
        list of tensors lol
    """
    with open("./datasets/coco.txt") as f:
        prompts = f.readlines()

    tokens = tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding=True)

    # Idgaf
    training_examples = []
    seq_lens = tokens["attention_mask"].sum(dim=1)
    for i in range(min(nsamples, len(prompts))):
        if seq_lens[i] <= seq_len:
            input_ids = tokens["input_ids"][i][: seq_lens[i]]
            training_examples.append(input_ids)

    return training_examples


def get_loaders(
    args,
    name,
    tokenizer,
    nsamples=128,
    seed=0,
    seqlen=2048,
    eval_mode=False,
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
            dataset = get_coco(nsamples, seqlen, tokenizer)

        with open(cached_dataset, "wb") as f:
            print(f"Saving cached tokenized dataset at {cached_dataset}...")
            pickle.dump(dataset, f)

    return dataset

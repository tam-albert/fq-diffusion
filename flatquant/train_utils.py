import gc
import itertools
import os
import time
from contextlib import nullcontext

import ipdb
import torch
import torch.nn as nn
from tqdm import tqdm

from flatquant import utils
from flatquant.function_utils import (
    check_params_grad,
    get_n_set_parameters_byname,
    get_paras_dict_by_name,
    set_require_grad_all,
)
from flatquant.model_utils import PIXART_LATENT_HEIGHT, PIXART_LATENT_WIDTH
from flatquant.quant_utils import set_quantizer_state


@torch.no_grad()
def prepare_calibration_data(args, pipe, data) -> list[dict[str, torch.Tensor]]:
    """
    Prepares calibration inputs at different timesteps (according to the scheduler),
    so that we can calibrate our model for quantization based on the results at
    different timesteps. Does this by just denoising.
    """

    calibration_data = []

    def callback(_, timestep, latents):
        prompt_timesteps.append(timestep)
        prompt_latents.append(latents)

    for prompt in tqdm(data, desc="Preparing calibration data..."):
        prompt_timesteps = []
        prompt_latents = []

        latents = torch.randn(
            (1, 4, PIXART_LATENT_HEIGHT, PIXART_LATENT_WIDTH),
            dtype=torch.float32,
            device=pipe.device,
        )

        prompt_latents.append(latents)

        prompt_embeds, prompt_attention_mask, *_ = pipe.encode_prompt(prompt)
        print(f"THE PROMPT LATENTS BEFORE PIPE HAVE LENGTH {len(prompt_latents)}")

        pipe(
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            num_inference_steps=args.cali_timesteps,
            guidance_scale=4.5,
            latents=latents,
            output_type="latent",
            callback=callback,
        )

        print(f"THE PROMPT LATENTS AFTER PIPE HAVE LENGTH {len(prompt_latents)}")

        # we don't need the fully denoised image (output) for calibration
        # this also ensures that the _inputs_ are lined up with the correct timesteps
        # since the callback is called with the _output_ for each timestep
        prompt_latents.pop()

        calibration_data.append(
            {
                "prompt_emb": prompt_embeds.cpu(),
                "prompt_attention_mask": prompt_attention_mask.cpu(),
                "timesteps": [timestep.cpu() for timestep in prompt_timesteps],
                "latents": [latent.cpu() for latent in prompt_latents],
            }
        )

    return calibration_data


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_value = None
        self.counter = 0
        self.stop = False

    def __call__(self, current_value):
        if self.best_value is None: # initialize to current value if first epoch
            self.best_value = current_value
            return
        if current_value < (self.best_value - self.min_delta):
            self.best_value = current_value
            self.counter = 0
        else: 
            self.counter += 1
            print(f"No improvement for {self.counter}/{self.patience} epochs")
            if self.counter >= self.patience:
                print("Stopping training")
                self.stop = True

def cali_flat_quant(args, pipe, calibration_data, dev, logger):
    # check trainable parameters
    for name, param in pipe.transformer.named_parameters():
        param.requires_grad = False

    # activate AMP
    if args.deactive_amp:
        dtype = torch.float32
        traincast = nullcontext
    else:
        dtype = torch.float16
        traincast = torch.cuda.amp.autocast

    ###
    # when calibrating, we only care about the part of the network that consists
    # of the transformer blocks. but there is a ton of preprocessing before that
    # (embedding the prompt, additional projection layers, positional embedding,
    # etc.)

    # but we only care about the input to the first transformer block. the original
    # code does this: just interrupt execution after the preprocessing is done, and
    # capture these inputs

    blocks = pipe.transformer.transformer_blocks

    hidden_states_cache = []
    encoder_hidden_states_cache = []
    encoder_attention_masks_cache = []
    timestep_embeddings_cache = []

    class Capturer(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, hidden_states, **kwargs):
            hidden_states_cache.append(hidden_states)
            encoder_hidden_states_cache.append(kwargs["encoder_hidden_states"])
            encoder_attention_masks_cache.append(kwargs["encoder_attention_mask"])
            timestep_embeddings_cache.append(kwargs["timestep"])
            raise ValueError

    blocks[0] = Capturer(blocks[0])

    with torch.no_grad():
        for item in calibration_data:
            prompt_emb = item["prompt_emb"].to(dev)
            prompt_attention_mask = item["prompt_attention_mask"].to(dev)

            for timestep, latent in zip(item["timesteps"], item["latents"]):
                latent = latent.to(dev)
                timestep = timestep.to(dev)

                try:
                    pipe(
                        prompt_embeds=prompt_emb,
                        prompt_attention_mask=prompt_attention_mask,
                        num_inference_steps=1,
                        guidance_scale=4.5,
                        latents=latent,
                        timestep=timestep,
                        output_type="latent",
                    )
                except ValueError:
                    pass

    hidden_states_cache = torch.cat(hidden_states_cache, dim=0)
    encoder_hidden_states_cache = torch.cat(encoder_hidden_states_cache, dim=0)
    encoder_attention_masks_cache = torch.cat(encoder_attention_masks_cache, dim=0)
    timestep_embeddings_cache = torch.cat(timestep_embeddings_cache, dim=0)

    blocks[0] = blocks[0].module

    # done capturing inputs, now we can start training the affine transforms

    # free up VRAM so we can train the affine transforms
    # we don't need the text encoder or the VAE, so get rid of them
    # otherwise, this will OOM when we move it to the CPU lol

    current = torch.cuda.memory_allocated() / 1024**2
    peak = torch.cuda.max_memory_allocated() / 1024**2
    print(f"Current GPU memory: {current:.2f}MB, Peak: {peak:.2f}MB")

    # pipe.text_encoder = None
    # pipe.vae = None
    pipe.to("cpu")

    current = torch.cuda.memory_allocated() / 1024**2
    peak = torch.cuda.max_memory_allocated() / 1024**2
    print(f"Current GPU memory: {current:.2f}MB, Peak: {peak:.2f}MB")

    # for each block, we want to track the inputs/outputs from the full-precision model
    fp_inps = hidden_states_cache  # initialize

    loss_fn = torch.nn.MSELoss()

    # begin training each block
    flat_parameters = {}
    
    for i, block in enumerate(blocks):
        # can change this to loop over just first block to make testing faster
        if i > -1:
            logger.info(f"========= Block {i} =========")
            current = torch.cuda.memory_allocated() / 1024**2
            peak = torch.cuda.max_memory_allocated() / 1024**2
            print(f"Current GPU memory: {current:.2f}MB, Peak: {peak:.2f}MB")
            dtype_dict = {}
            block = block.to(dev)

            current = torch.cuda.memory_allocated() / 1024**2
            print(f"new current GPU memory: {current:.2f}MB")

            for name, param in block.named_parameters():
                dtype_dict[name] = param.dtype

            with torch.no_grad():
                block.float()  # run in full-precision

            # run the layer in full precision and save the outputs
            fp_outs = []

            # block.self_attn._ori_mode = True
            block.attn1._ori_mode = True
            block.attn2._ori_mode = True
            block.ff._ori_mode = True
            with torch.no_grad():
                for j in tqdm(
                    range(args.nsamples * args.cali_timesteps),
                    desc="collecting block outputs...",
                ):
                    fp_outs.append(
                        block(
                            hidden_states=fp_inps[j : j + 1],
                            encoder_hidden_states=encoder_hidden_states_cache[j : j + 1],
                            encoder_attention_mask=encoder_attention_masks_cache[j : j + 1],
                            timestep=timestep_embeddings_cache[j : j + 1],
                        )
                    )

            fp_outs = torch.cat(fp_outs, dim=0)
            # block.self_attn._ori_mode = False
            block.attn1._ori_mode = False
            block.attn2._ori_mode = False
            block.ff._ori_mode = False

            # initialize per-channel smoothing factor (SmoothQuant)
            if args.diag_init == "sq_style":
                # block.self_attn.init_diag_scale(alpha=args.diag_alpha)
                block.attn1.init_diag_scale(alpha=args.diag_alpha)
                block.attn2.init_diag_scale(alpha=args.diag_alpha)
                block.ff.init_diag_scale(alpha=args.diag_alpha)
            elif args.diag_init == "one_style":
                pass
            else:
                raise NotImplementedError

            # begin learning affine transforms for this block
            block = block.to(dev)
            set_require_grad_all(block, False)

            param_configs = {
                "cali_trans": {"name": "trans.linear", "lr_multiplier": 1},
                "add_diag": {"name": "trans.diag_scale", "lr_multiplier": 1},
                "lwc": {"name": "clip_factor_w", "lr_multiplier": 3},
                "lac": {"name": "clip_factor_a", "lr_multiplier": 3},
            }

            trained_params = []
            paras_name = []

            for arg_name, config in param_configs.items():
                if getattr(args, arg_name):
                    trained_params.append(
                        {
                            "params": get_n_set_parameters_byname(block, [config["name"]]),
                            "lr": args.flat_lr * config["lr_multiplier"],
                        }
                    )
                    paras_name.append(config["name"])

            print(f"{trained_params=}")
            print(f"{paras_name=}")

            optimizer = torch.optim.AdamW(trained_params)
            scheduler_main = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.epochs * (args.nsamples * args.cali_timesteps // args.cali_bsz),
                eta_min=args.flat_lr * 1e-3,
            )
            if args.warmup:
                scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.01, total_iters=16
                )
                scheduler = torch.optim.lr_scheduler.ChainedScheduler(
                    [scheduler_warmup, scheduler_main]
                )
            else:
                scheduler = scheduler_main
            # check_params_grad(layer)
            # set_quantizer_state(layer, False)

            early_stopper = EarlyStopping(args.patience)



            for epoch in range(args.epochs):
                #current = torch.cuda.memory_allocated() / 1024**2
                #print(f"GPU memory at start of epoch {epoch}: {current:.2f}MB")
                mse = 0
                start_tick = time.time()
                with traincast():
                    for batch_i in range(
                        args.nsamples * args.cali_timesteps // args.cali_bsz
                    ):
                        index = batch_i * args.cali_bsz
                        quant_out = block(
                            hidden_states=fp_inps[index : index + args.cali_bsz],
                            encoder_hidden_states=encoder_hidden_states_cache[
                                index : index + args.cali_bsz
                            ],
                            encoder_attention_mask=encoder_attention_masks_cache[
                                index : index + args.cali_bsz
                            ],
                            timestep=timestep_embeddings_cache[
                                index : index + args.cali_bsz
                            ],
                        )
                        #current = torch.cuda.memory_allocated() / 1024**2
                        #print(f"GPU memory before backprop in epoch {epoch}: {current:.2f}MB")
                        loss = loss_fn(fp_outs[index : index + args.cali_bsz], quant_out)
                        #current = torch.cuda.memory_allocated() / 1024**2
                        #print(f"GPU memory after backprop in epoch {epoch}: {current:.2f}MB")
                        mse += loss.detach().cpu()
                        loss = loss / loss.clone().detach()
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                        
                cur_lr = optimizer.state_dict()["param_groups"][0]["lr"]
                logger.info(
                    f"block {i} lwc lac iter {epoch}, lr {cur_lr:.8f}  time {time.time() - start_tick:.6f}s, mse: {mse:.8f}"
                )

                early_stopper(mse)
                if early_stopper.stop:
                    logger.info("=============Early stopping============")
                    break

            fp_inps, fp_outs = fp_outs, fp_inps
            blocks[i] = block.to("cpu")
            flat_parameters[i] = get_paras_dict_by_name(block, required_names=paras_name)
            torch.save(flat_parameters, os.path.join(args.exp_dir, f"flat_parameters.pth"))
            logger.info(
                "saved paramaters at {}".format(
                    os.path.join(args.exp_dir, f"flat_parameters.pth")
                )
            )
            for name, param in block.named_parameters():
                param.requires_grad = False
                if name in dtype_dict.keys():
                    param.data = param.to(dtype_dict[name])
            del block
            torch.cuda.empty_cache()

    del fp_inps, fp_outs
    gc.collect()
    torch.cuda.empty_cache()
    return pipe

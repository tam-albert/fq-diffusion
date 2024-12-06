import gc
import os
import time
from contextlib import nullcontext

import torch
import torch.nn as nn

from flatquant import utils
from flatquant.function_utils import (
    check_params_grad,
    get_n_set_parameters_byname,
    get_paras_dict_by_name,
    set_require_grad_all,
)
from flatquant.quant_utils import set_quantizer_state


# def prepare_calibration_inputs(args, model, scheduler, dataloader, dev)
#     """
#     Prepares calibration inputs for the whole model.
#     """


def cali_flat_quant(args, model, scheduler, dataloader, dev, logger):
    model.eval()

    # check trainable parameters
    for name, param in model.named_parameters():
        param.requires_grad = False

    # activate AMP
    if args.deactive_amp:
        dtype = torch.float32
        traincast = nullcontext
    else:
        dtype = torch.float16
        traincast = torch.cuda.amp.autocast

    # prepare_calibration_inputs(args, model, scheduler, dataloader, dev)

    ## in this section, we will call the model to get the input of the first layer
    # we will interrupt execution so the model does all the preprocessing for us
    # super sus

    # result of interrupting execution

    torch.manual_seed(0)
    latent = torch.randn((1, 4, 128, 128), dtype=dtype, device=dev)

    # start
    # Prepare timesteps
    scheduler.set_timesteps(args.cali_timesteps, device=dev)
    timesteps = scheduler.timesteps
    # DELETE:
    timesteps = torch.tensor([0], dtype=torch.long, device=dev)

    model.adaln_single.to(dev)
    timesteps, _ = model.adaln_single(
        timesteps,
        added_cond_kwargs={"resolution": None, "aspect_ratio": None},
        hidden_dtype=dtype,
    )
    model.adaln_single.cpu()

    # Prepare model inputs (n_samples x num_timesteps)
    model.caption_projection.to(dev)
    with torch.no_grad():
        for prompt_emb, prompt_seq_len in dataloader:
            # Prepare

            prompt_emb = prompt_emb.to(dev)  # (bsz, max_seq_len, hidden_size)
            prompt_seq_len = prompt_seq_len.to(dev)  # (bsz,)

            B, L, C = prompt_emb.size()

            attention_mask = (
                torch.arange(L, device=dev).expand(B, -1) < prompt_seq_len.unsqueeze(1)
            ).to(dev)

            encoder_hidden_states = model.caption_projection(prompt_emb)

            model.pos_embed.to(dev)
            model.transformer_blocks[0].to(dev)

            print("Given latents:", latent)
            print("latent min:", latent.min())
            print("latent max:", latent.max())
            print("latent shape:", latent.size())
            print("latent sum:", latent.sum())

            inp = model.pos_embed(latent)
            inp = model.transformer_blocks[0](
                inp,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=attention_mask,
                timestep=timesteps,
            )

            print("input:", inp)
            print("input min:", inp.min())
            print("input max:", inp.max())
            print("input shape:", inp.size())
            print("input sum:", inp.sum())
            import ipdb

            ipdb.set_trace()

            model(
                latent,
                encoder_hidden_states=prompt_emb.to(dev),
                encoder_attention_mask=attention_mask,
                timestep=torch.tensor([0], dtype=torch.long, device=dev),  # TODO: fix,
                added_cond_kwargs={"resolution": None, "aspect_ratio": None},
                return_dict=False,
            )
    # end

    model.pos_embed.to(dev)
    model.transformer_blocks[0].to(dev)
    model.adaln_single.to(dev)
    model.caption_projection.to(dev)

    class Interrupter(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            print("Interrupting execution")
            print("input:", inp)
            print("input min:", inp.min())
            print("input max:", inp.max())
            print("input shape:", inp.size())
            print("input sum:", inp.sum())
            raise ValueError

    model.transformer_blocks[1] = Interrupter(model.transformer_blocks[1])
    # model.transformer_blocks[0] = Interrupter(model.transformer_blocks[0])

    with torch.no_grad():
        for prompt_emb, prompt_seq_len in dataloader:
            prompt_emb.to(dev)  # (bsz, max_seq_len, hidden_size)
            prompt_seq_len.to(dev)  # (bsz,)

            B, L, C = prompt_emb.size()

            attention_mask = (
                torch.arange(L).expand(B, -1) < prompt_seq_len.unsqueeze(1)
            ).to(dev)

            try:
                print("Given latents:", latent)
                print("latent min:", latent.min())
                print("latent max:", latent.max())
                print("latent shape:", latent.size())
                print("latent sum:", latent.sum())
                model(
                    latent,
                    encoder_hidden_states=prompt_emb.to(dev),
                    encoder_attention_mask=attention_mask,
                    timestep=torch.tensor(
                        [0], dtype=torch.long, device=dev
                    ),  # TODO: fix,
                    added_cond_kwargs={"resolution": None, "aspect_ratio": None},
                    return_dict=False,
                )
            except ValueError:
                pass

            break

    import ipdb

    ipdb.set_trace()

    # move embedding layer and first layer to target device
    blocks = model.transformer_blocks
    layers[0] = layers[0].to(dev)
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(dev)

    # catch the first layer input
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                sample = batch[0]
                model(sample.to(dev))
            except ValueError:
                pass
    position_ids = cache["position_ids"]
    attention_mask = cache["attention_mask"]
    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(args.cali_bsz, 1, 1, 1).float()
    else:
        attention_mask_batch = None

    # move embedding layer and first layer to cpu
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    # raise ValueError("Only support for llama-2/Llama-3/qwen-2 now")
    torch.cuda.empty_cache()

    #

    # same input of first layer for fp model and quant model
    fp_inps = inps  # take output of fp model as input
    fp_outs = torch.zeros_like(inps)  # take output of fp model as input

    loss_func = torch.nn.MSELoss()
    # start training
    flat_parameters = {}
    num_train_layer = len(layers)
    mse_dict = {}
    for i in range(num_train_layer):
        logger.info(f"========= Layer {i} =========")
        dtype_dict = {}
        layer = layers[i].to(dev)
        for name, param in layer.named_parameters():
            dtype_dict[name] = param.dtype
        with torch.no_grad():
            layer.float()

        # run the layer in full precision and save the outputs
        layer.self_attn._ori_mode = True
        layer.mlp._ori_mode = True
        with torch.no_grad():
            for j in range(args.nsamples):
                fp_outs[j] = layer(
                    fp_inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]
        layer.self_attn._ori_mode = False
        layer.mlp._ori_mode = False

        # initialize per-channel smoothing factor (SmoothQuant)
        if args.diag_init == "sq_style":
            layer.self_attn.init_diag_scale(alpha=args.diag_alpha)
            layer.mlp.init_diag_scale(alpha=args.diag_alpha)
        elif args.diag_init == "one_style":
            pass
        else:
            raise NotImplementedError

        # begin learning affine transforms for this layer
        layer = layer.to(dev)
        set_require_grad_all(layer, False)
        trained_params, paras_name = [], []
        if args.cali_trans:
            trained_params.append(
                {
                    "params": get_n_set_parameters_byname(
                        layer,
                        [
                            "trans.linear",
                        ],
                    ),
                    "lr": args.flat_lr,
                }
            )
            paras_name.append("trans.linear")
        if args.add_diag:
            trained_params.append(
                {
                    "params": get_n_set_parameters_byname(
                        layer,
                        [
                            "trans.diag_scale",
                        ],
                    ),
                    "lr": args.flat_lr,
                }
            )
            paras_name.append("trans.diag_scale")
        if args.lwc:
            trained_params.append(
                {
                    "params": get_n_set_parameters_byname(
                        layer,
                        [
                            "clip_factor_w",
                        ],
                    ),
                    "lr": args.flat_lr * 10,
                }
            )
            paras_name.append("clip_factor_w")
        if args.lac:
            trained_params.append(
                {
                    "params": get_n_set_parameters_byname(
                        layer,
                        [
                            "clip_factor_a",
                        ],
                    ),
                    "lr": args.flat_lr * 10,
                }
            )
            paras_name.append("clip_factor_a")

        optimizer = torch.optim.AdamW(trained_params)
        scheduler_main = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs * (args.nsamples // args.cali_bsz),
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
        for epoch in range(args.epochs):
            mse = 0
            start_tick = time.time()
            with traincast():
                for j in range(args.nsamples // args.cali_bsz):
                    index = j * args.cali_bsz
                    quant_out = layer(
                        fp_inps[index : index + args.cali_bsz,],
                        attention_mask=attention_mask_batch,
                        position_ids=position_ids,
                    )[0]
                    loss = loss_func(fp_outs[index : index + args.cali_bsz,], quant_out)
                    mse += loss.detach().cpu()
                    loss = loss / loss.clone().detach()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
            cur_lr = optimizer.state_dict()["param_groups"][0]["lr"]
            logger.info(
                f"layer {i} lwc lac iter {epoch}, lr {cur_lr:.8f}  time {time.time() - start_tick:.6f}s, mse: {mse:.8f}"
            )

        fp_inps, fp_outs = fp_outs, fp_inps
        layers[i] = layer.to("cpu")
        flat_parameters[i] = get_paras_dict_by_name(layer, required_names=paras_name)
        torch.save(flat_parameters, os.path.join(args.exp_dir, f"flat_parameters.pth"))
        logger.info(
            "saved paramaters at {}".format(
                os.path.join(args.exp_dir, f"flat_parameters.pth")
            )
        )
        for name, param in layer.named_parameters():
            param.requires_grad = False
            if name in dtype_dict.keys():
                param.data = param.to(dtype_dict[name])
        del layer
        torch.cuda.empty_cache()

    del inps, fp_inps, fp_outs
    gc.collect()
    torch.cuda.empty_cache()
    return model

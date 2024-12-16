import os
import torch
from flatquant.function_utils import get_paras_dict_by_name
import logging


def kronecker_matmul(x, hadL, hadR):
    """equivalent to

    had = torch.kron(hadL, hadR)
    x = x.reshape(-1, had.shape[0])
    x = x.matmul(had).reshape(init_shape)
    """
    init_shape = x.shape
    x = x.reshape(-1, hadL.shape[0], hadR.shape[0])
    x = torch.matmul(x, hadR)
    x = torch.matmul(hadL.T, x)
    return x.reshape(init_shape)


def reparameterize_ln(ln, trans):
    # assert isinstance(ln, (LlamaRMSNorm, Qwen2RMSNorm))
    ln_weight = ln.weight.data
    ori_dtype = ln_weight.dtype
    ln_weight = ln_weight.to(torch.float64)
    ln_weight = ln_weight * trans.diag_scale.to(torch.float64)
    ln.weight.data = ln_weight.to(ori_dtype)
    trans.use_diag = False


def reparameterize_model(model, is_pixart = False):
    if is_pixart:
        for idx in range(len(model.transformer_blocks)):
            layer = model.transformer_blocks[idx]
            layer.attn1.reparameterize()
            layer.attn2.reparameterize()
            #layer.ff.reparameterize()
            # fuse per-channel scaling to layernorm
            # if layer.attn1.ln_trans is not None and layer.attn1.ln_trans.add_diag:
            #     reparameterize_ln(layer.input_layernorm, layer.attn1.ln_trans)
            # if layer.attn2.ln_trans is not None and layer.attn2.ln_trans.add_diag:
            #     reparameterize_ln(layer.input_layernorm, layer.attn2.ln_trans)
    else:
        for idx in range(model.config.num_hidden_layers):
            layer = model.model.layers[idx]
            layer.self_attn.reparameterize()
            layer.mlp.reparameterize()
            # fuse per-channel scaling to layernorm
            if layer.self_attn.ln_trans is not None and layer.self_attn.ln_trans.add_diag:
                reparameterize_ln(layer.input_layernorm, layer.self_attn.ln_trans)
            if layer.mlp.up_gate_trans is not None and layer.mlp.up_gate_trans.add_diag:
                reparameterize_ln(layer.post_attention_layernorm, layer.mlp.up_gate_trans)
    return model


def save_parametrized_checkpoint(model, args, is_pixart = True):
    if is_pixart:
        quanted_parameters = {}
        for i in range(len(model.transformer_blocks)):
            layer = model.transformer_blocks[i]
            quanted_parameters[i] = layer.state_dict()
        torch.save(
            quanted_parameters, os.path.join(args.exp_dir, f"parametrized_paras.pth")
        )
        logging.info(
            "saved paramaters at {}".format(
                os.path.join(args.exp_dir, f"parametrized_paras.pth")
            )
        )
    else:
        quanted_parameters = {}
        for i in range(len(model.model.layers)):
            layer = model.model.layers[i]
            quanted_parameters[i] = layer.state_dict()
        torch.save(
            quanted_parameters, os.path.join(args.exp_dir, f"parametrized_paras.pth")
        )
        logging.info(
            "saved paramaters at {}".format(
                os.path.join(args.exp_dir, f"parametrized_paras.pth")
            )
        )


def load_flat_parameters(args, model, path=None, is_pixart = True):
    if is_pixart:
        if path is None:
            print(str(os.path.join(args.exp_dir, f"flat_parameters.pth")))
            flat_parameters = torch.load(os.path.join(args.exp_dir, f"flat_parameters.pth"))
        else:
            flat_parameters = torch.load(os.path.join(path, f"flat_parameters.pth"))
        layers = model.transformer_blocks

        for i in range(len(flat_parameters.keys())):
            flat_param = flat_parameters[i]
            layers[i].load_state_dict(flat_param, strict=False)
        return model
    else:
        if path is None:
            flat_parameters = torch.load(os.path.join(args.exp_dir, f"flat_parameters.pth"))
        else:
            flat_parameters = torch.load(os.path.join(path, f"flat_parameters.pth"))
        layers = model.model.layers

        for i in range(len(flat_parameters.keys())):
            flat_param = flat_parameters[i]
            layers[i].load_state_dict(flat_param, strict=False)
        return model


def save_flat_matrices(args, model, is_pixart = False):
    flat_matrices = {}
    if is_pixart:
        for i in range(len(model.transformer_blocks)):
            layer = model.transformer_blocks[i]
            #layer.attn1.rep_matrix_only()
            #layer.attn2.rep_matrix_only()
            #layer.ff.rep_matrix_only()
            paras_name = [
                "trans.matrix",
                "trans.diag_scale",
                "clip_factor_w",
                "clip_factor_a",
            ]
            flat_matrices[i] = get_paras_dict_by_name(layer, required_names=paras_name)
        torch.save(flat_matrices, os.path.join(args.exp_dir, f"flat_matrices.pth"))
        logging.info(
            "saved paramaters at {}".format(
                os.path.join(args.exp_dir, f"flat_matrices.pth")
            )
        )
    else:
        for i in range(len(model.model.layers)):
            layer = model.model.layers[i]
            layer.self_attn.rep_matrix_only()
            layer.mlp.rep_matrix_only()
            paras_name = [
                "trans.matrix",
                "trans.diag_scale",
                "clip_factor_w",
                "clip_factor_a",
            ]
            flat_matrices[i] = get_paras_dict_by_name(layer, required_names=paras_name)
        torch.save(flat_matrices, os.path.join(args.exp_dir, f"flat_matrices.pth"))
        logging.info(
            "saved paramaters at {}".format(
                os.path.join(args.exp_dir, f"flat_matrices.pth")
            )
        )


def load_flat_matrices(args, model, path=None, is_pixart = False):
    if path is None:
        flat_parameters = torch.load(os.path.join(args.exp_dir, f"flat_matrices.pth"))
    else:
        flat_parameters = torch.load(os.path.join(path, f"flat_matrices.pth"))

    if is_pixart:
        layers = model.transformer_blocks
        for i in range(len(flat_parameters.keys())):
            flat_param = flat_parameters[i]
            print("WE ARE GETTING INTO LOAD FLAT MATRICES")
            #print(f"ATTN1 ORI MODE IS {layers[i].attn1._ori_mode}")
            #print(f"ATTN1 ORI MODE IS {layers[i].attn1._ori_mode}")

            #layers[i].attn1.rep_matrix_only()
            #layers[i].attn2.rep_matrix_only()
            #layers[i].ff.rep_matrix_only()
            layers[i].load_state_dict(flat_param, strict=False)
        return model

    else:
        layers = model.model.layers

        for i in range(len(flat_parameters.keys())):
            flat_param = flat_parameters[i]
            layers[i].self_attn.rep_matrix_only()
            layers[i].mlp.rep_matrix_only()
            layers[i].load_state_dict(flat_param, strict=False)
        return model

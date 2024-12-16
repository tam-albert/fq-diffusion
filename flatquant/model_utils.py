import logging

import diffusers
import torch
import transformers

from flatquant.model_tools.llama31_utils import apply_flatquant_to_llama_31
from flatquant.model_tools.llama_utils import apply_flatquant_to_llama
from flatquant.model_tools.pixart_utils import apply_flatquant_to_pixart
from flatquant.utils import skip

PIXART_IMAGE_HEIGHT = 1024
PIXART_IMAGE_WIDTH = 1024
PIXART_LATENT_HEIGHT = PIXART_IMAGE_HEIGHT // 8
PIXART_LATENT_WIDTH = PIXART_IMAGE_WIDTH // 8


def skip_initialization():
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip


def get_llama(model_name, hf_token):
    skip_initialization()
    config = transformers.LlamaConfig.from_pretrained(model_name)
    config._attn_implementation_internal = "eager"
    model = transformers.LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        config=config,
        use_auth_token=hf_token,
        low_cpu_mem_usage=True,
    )
    model.seqlen = 2048
    logging.info(f"---> Loading {model_name} Model with seq_len: {model.seqlen}")
    return model, apply_flatquant_to_llama


def get_llama_31(model_name, hf_token):
    skip_initialization()
    config = transformers.LlamaConfig.from_pretrained(model_name)
    config._attn_implementation_internal = "eager"
    model = transformers.LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        config=config,
        use_auth_token=hf_token,
        low_cpu_mem_usage=True,
    )
    model.seqlen = 2048
    logging.info(f"---> Loading {model_name} Model with seq_len: {model.seqlen}")
    return model, apply_flatquant_to_llama_31


def get_qwen2(model_name, hf_token):
    skip_initialization()
    try:
        from transformers import Qwen2ForCausalLM
    except ImportError:
        logging.error(
            "Qwen2 model is not available in this version of 'transformers'. Please update the library."
        )
        raise ImportError(
            "Qwen2 model is not available. Ensure you're using a compatible version of the 'transformers' library."
        )

    config = transformers.Qwen2Config.from_pretrained(model_name)
    config._attn_implementation_internal = "eager"
    model = Qwen2ForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        config=config,
        use_auth_token=hf_token,
        low_cpu_mem_usage=True,
    )
    model.seqlen = 2048
    logging.info(f"---> Loading {model_name} Model with seq_len: {model.seqlen}")

    from flatquant.model_tools.qwen_utils import apply_flatquant_to_qwen

    return model, apply_flatquant_to_qwen


def get_opt(model_name):
    skip_initialization()
    model = transformers.OPTForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", low_cpu_mem_usage=True
    )
    model.seqlen = model.config.max_position_embeddings
    logging.info(f"---> Loading {model_name} Model with seq_len: {model.seqlen}")
    raise NotImplementedError("Post-processing for OPT model is not implemented yet.")


def get_pixart(model_name):
    print(model_name)
    skip_initialization()

    pipe = diffusers.PixArtSigmaPipeline.from_pretrained(
        model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )

    pipe.transformer.seqlen = 512  # T5

    return pipe, apply_flatquant_to_pixart


# Unified model loading function
def get_model(model_name):
    if "pixart-sigma" in model_name.lower():
        return get_pixart(model_name)
    else:
        raise ValueError(f"Unknown model {model_name}")

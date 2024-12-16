import flatquant.utils as utils
import flatquant.args_utils as args_utils
import flatquant.model_utils as model_utils
import flatquant.data_utils as data_utils
import flatquant.eval_utils as eval_utils
import flatquant.train_utils as train_utils
import flatquant.flat_utils as flat_utils
import gptq_utils

import torch
from transformers import AutoModel, AutoTokenizer
from diffusers import PixArtAlphaPipeline

def get_detailed_model_size(model):
    total_params = 0
    trainable_params = 0
    
    for name, parameter in model.named_parameters():
        param_count = parameter.nelement()
        total_params += param_count
        if parameter.requires_grad:
            trainable_params += param_count
        
        print(f"{name}: {param_count} params, {param_count * parameter.element_size() / 1024**2:.2f} MB")
    
    print(f"\nTotal params: {total_params}")
    print(f"Trainable params: {trainable_params}")
    print(f"Total model size: {total_params * parameter.element_size() / 1024**2:.2f} MB")


def main():
    # args, logger = args_utils.parser_gen()
    # utils.seed_everything(seed=args.seed)
    # print(args)

    # pipe, apply_flatquant_to_model = model_utils.get_model(args.model)
    # pipe.to(utils.DEV)

    # model = pipe.transformer

    # print(model.transformer_blocks)

    # # Assuming 'pipe' is your model
    # get_detailed_model_size(pipe.transformer)
    pipe = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS", torch_dtype=torch.float16)
    pipe = pipe.to('cuda')
    with torch.no_grad():
        input_text = "A tennis player killing dragons"
        print(f"Input text: {input_text}")
        pipe.to('cuda')
        generated_image_path = "generated_image_baseline_pixart.png"
        print(f"Saving generated image at {generated_image_path}.")
        output = pipe(input_text)
        output.images[0].save(generated_image_path, format="PNG")
        print("Image has been saved!")

    return
    

if __name__ == '__main__':
    main()
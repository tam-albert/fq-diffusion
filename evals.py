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
from datasets import load_dataset

from torchmetrics.image.fid import FrechetInceptionDistance
from pytorch_fid import fid_score
import time
import random
from itertools import cycle

import json
import os

random.seed(1)

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
    args, logger = args_utils.parser_gen()
    utils.seed_everything(seed=args.seed)
    print(args)

    pipe, apply_flatquant_to_model = model_utils.get_model(args.model)
    pipe.to(utils.DEV)

    model = pipe.transformer

    print(model.transformer_blocks)

    # Assuming 'pipe' is your model
    get_detailed_model_size(pipe.transformer)

    # get calibration data
    data = data_utils.get_loaders(
        args,
        args.cali_dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        eval_mode=False,
    )

    """
    pipe.transformer = apply_flatquant_to_model(args, pipe.transformer)
    logger.info("Finished applying FlatQuant to model.")
    """

    if args.quantize:
        # 1. prepare calibration data
        if args.cali_trans or args.add_diag or args.lwc or args.lac:
            calibration_data = train_utils.prepare_calibration_data(args, pipe, data)
            logger.info("Finished preparing calibration data.")
        
        # 2. replace linear/attention layers with special FlatQuant layers
        pipe.transformer = apply_flatquant_to_model(args, pipe.transformer, shared = False)
        logger.info("Finished applying FlatQuant to model.")

        if args.resume:
            print("testing did it get here?")
            flat_utils.load_flat_parameters(args, model, is_pixart = True)
        if args.reload_matrix:
            print("testing did it get to flat matrices?")
            flat_utils.load_flat_matrices(args, model, path=args.matrix_path, is_pixart = True)
        elif args.cali_trans or args.add_diag or args.lwc or args.lac:
            # 2. calibrate FlatQuant layers (learn affine transforms)
            train_utils.cali_flat_quant(
                args, pipe, calibration_data, utils.DEV, logger=logger
            )
     
        if args.save_matrix and not args.reload_matrix:
            flat_utils.save_flat_matrices(args, model, is_pixart = True)
            print("Finished saving the model")
        #flat_utils.reparameterize_model(model.to('cuda'), is_pixart = True)
        #logger.info("Finished reparameterize model.")

    # 3. actually quantize the weights
    print(pipe)

    # SAMPLE INFERENCE
    pipe.to('cuda')
    """
    with torch.no_grad():
        #input_text = "A tennis player killing dragons"
        input_prompts = [
            "Dinosaur doing a heel hook",
            "Tennis player killing dragons",
            "really fancy fashion brand",
            "cool new computer",
            "Really large concert",
            "Travelling to Italy",

        ]
        input_text = "Dinosaur doing a heel hook"
        print(f"Input text: {input_text}")
        pipe.to('cuda')
        start_time = time.time()
        img_path = "generated_image_w4a8_30timesteps.png"
        print(f"Saving generated image at {img_path}.")
        output = pipe(input_text)
        output.images[0].save(img_path, format="PNG")
        print("Image has been saved!")
        print(time.time() - start_time)
    """
    """
    EVAL CODE
    """

    if not args.eval:
        return

    print(f"Running inference on MHJQ prompts for {args.model} w{args.w_bits}a{args.a_bits}, k{args.k_bits}v{args.v_bits} quantization.")
    print(f"EXP DIR: {args.exp_dir}")

    
    mjhq_json_path = "/home/user/fq-diffusion/mjhq_dataset/metadata.json"
    with open(mjhq_json_path, "r") as file:
        data = json.load(file)

    # Convert to the desired structure
    category_dict = {}

    for item_id, item_data in data.items():
        category = item_data["category"]
        prompt = item_data["prompt"]
        
        # Add to the category dictionary
        if category not in category_dict:
            category_dict[category] = []
        category_dict[category].append((category, prompt, item_id))

    print("Number of images for each category:", {k: len(v) for k, v in category_dict.items()})

    # Save file paths
    generated_images_path = f"{args.exp_dir}/generated_images_new"
    if not os.path.exists(generated_images_path):
        os.makedirs(generated_images_path)
    # for category in category_dict:
    #     if not os.path.exists(f"{generated_images_path}/{category}"):
    #         os.makedirs(f"{generated_images_path}/{category}")
    print("Created directory", os.listdir(generated_images_path))


    # Alternating between categories
    result_list = [item for group in zip(*[category_list for _, category_list in category_dict.items()]) for item in group]

    def batch_generator(lst, batch_size):
        for i in range(0, len(lst), batch_size):
            yield lst[i:i + batch_size]

    BATCH_SIZE = 1

    i = 0
    for batch in batch_generator(result_list, BATCH_SIZE):
        start_time = time.time()

        categories, prompts, img_ids = [x[0] for x in batch], [x[1] for x in batch], [x[2] for x in batch]
        save_location = f"{generated_images_path}/{img_ids[0]}.png"
        if os.path.exists(save_location):
            # we've already generated this sample. skip it
            continue

        print("CATEGORIES")
        print("-----------------")
        print(categories)

        print("PROMPTS")
        print("------------------")
        print(prompts)

        print("IDS")
        print("-----------------")
        print(img_ids)

        utils.seed_everything(seed=args.seed) 
        output = pipe(prompts)
        for i in range(BATCH_SIZE):
            output.images[i].save(save_location, format="PNG")

        print(f"{time.time() - start_time} taken on batch of size {BATCH_SIZE}.")


    return
    fid_value = fid_score.calculate_fid_given_paths([real_images_path, generated_images_path], 
                                                batch_size=50, 
                                                device='cuda', 
                                                dims=2048)
    print(f"FID Score: {fid_value}")


if __name__ == "__main__":
    main()

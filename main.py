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

from torchmetrics.image.fid import FrechetInceptionDistance
from pytorch_fid import fid_score

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

    
    #pipe.transformer = apply_flatquant_to_model(args, pipe.transformer)
    logger.info("Finished applying FlatQuant to model.")
    
    if args.quantize:
        # 1. prepare calibration data
        if args.eval or args.cali_trans or args.add_diag or args.lwc or args.lac:
            calibration_data = train_utils.prepare_calibration_data(args, pipe, data)
            logger.info("Finished preparing calibration data.")
        
        # 2. replace linear/attention layers with special FlatQuant layers
        pipe.transformer = apply_flatquant_to_model(args, pipe.transformer, shared = True)
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
    # SAMPLE INFERENCE

    # input_text = "medium rare steak tenderloin super tasty photo"
    # input_text = "a dragon soaring through the skies, above mountainous terrain"
    input_text = "a dragon soaring through the skies, above mountainous terrain"
    utils.seed_everything(seed=args.seed)
    with torch.no_grad():
        print(f"Input text: {input_text}")
        pipe.to('cuda')
        generated_image_path = "generated_image_trained_quant_w4a8.png"
        print(f"Saving generated image at {generated_image_path}.")
        output = pipe(input_text)
        output.images[0].save(generated_image_path, format="PNG")
        print("Image has been saved!")

    return


    """
    EVAL CODE
    """

    def compute_fid_score(generated_image_path, target_image_path):

        fid = FrechetInceptionDistance(feature=2048) 

        transform = transforms.Compose([
            transforms.Resize((299, 299)),  # Inception expects 299x299
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        def load_and_process_image(image_path):
            image = default_loader(image_path)  
            return transform(image).unsqueeze(0) 

        generated_image = load_and_process_image(generated_image_path)
        target_image = load_and_process_image(target_image_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fid = fid.to(device)
        generated_image = generated_image.to(device)
        target_image = target_image.to(device)

        fid.update(generated_image, real=False)  
        fid.update(target_image, real=True)    

        fid_score = fid.compute()
        return fid_score

    if args.eval:
        print(f"Running evaluations for {args.model} w{args.w_bits}a{args.a_bits}, k{args.k_bits}v{args.v_bits} quantization.")
    print(calibration_data) 
    
    # if args.distribute_model:
    #     utils.distribute_model(model)text_encoder.to(device)
    # else:
    #     model.to(utils.DEV)

    """
    # Evaluating PPL
    for eval_dataset in ["wikitext2", "c4"]:
        logger.info(eval_dataset)
        testloader = data_utils.get_loaders(
            args,
            eval_dataset,
            seed=args.seed,
            model=args.model,
            seqlen=model.seqlen,
            hf_token=args.hf_token,
            eval_mode=True,
        )
        dataset_ppl = eval_utils.ppl_eval(model, testloader)
        logger.info(dataset_ppl)

    if args.lm_eval:
        import lm_eval
        from lm_eval import utils as lm_eval_utils
        from lm_eval.models.huggingface import HFLM

        hflm = HFLM(
            pretrained=model, tokenizer=tokenizer, batch_size=args.lm_eval_batch_size
        )

        task_manager = lm_eval.tasks.TaskManager(
            include_path="./datasets/lm_eval_configs/tasks", include_defaults=False
        )
        task_names = lm_eval_utils.pattern_match(args.tasks, task_manager.all_tasks)
        results = {}
        for task_name in task_names:
            logger.info(f"Evaluating {task_name}...")
            result = lm_eval.simple_evaluate(
                hflm,
                tasks=[task_name],
                batch_size=args.lm_eval_batch_size,
                task_manager=task_manager,
            )["results"]
            result = result[task_name]
            acc = round(result.get("acc_norm,none", result["acc,none"]) * 100, 2)
            results[task_name] = acc
            logger.info(f"acc: {acc}%")
        metric_vals = {task: result for task, result in results.items()}
        metric_vals["acc_avg"] = round(
            sum(metric_vals.values()) / len(metric_vals.values()), 2
        )
        logger.info(metric_vals)
    """


if __name__ == "__main__":
    main()

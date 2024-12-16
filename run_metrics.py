
import lpips
from PIL import Image
import os
from torchvision import transforms
import shutil

MJ_HIERARCHICAL_DIR = "/home/user/fq-diffusion/mjhq_dataset" 
FP16_HIERARCHICAL_DIR = "/home/user/fq-diffusion/outputs/PixArt-Sigma-XL-2-1024-MS/w16a16/exp/generated_images"
W8A8_HIERARCHICAL_DIR = "/home/user/fq-diffusion/outputs/PixArt-Sigma-XL-2-1024-MS/w4a8/exp/generated_images"

MJ_DIR = "/home/user/fq-diffusion/mjhq_flattened"
FP16_DIR = "/home/user/fq-diffusion/outputs/PixArt-Sigma-XL-2-1024-MS/w16a16/exp/generated_images_new"
W8A8_DIR = "/home/user/fq-diffusion/outputs/PixArt-Sigma-XL-2-1024-MS/w4a8/exp/generated_images_new"

CATEGORIES = ["animals",  "art",  "fashion",  "food",  "indoor",  "landscape",  "logo",  "people",  "plants",  "vehicles"]

if False:

    os.makedirs(MJ_DIR, exist_ok=True)
    os.makedirs(FP16_DIR, exist_ok=True)
    os.makedirs(W8A8_DIR, exist_ok=True)

    for category in CATEGORIES:

            category_original_path = os.path.join(MJ_HIERARCHICAL_DIR, category)
            category_w8a8_path = os.path.join(W8A8_HIERARCHICAL_DIR, category)
            category_fp16_path = os.path.join(FP16_HIERARCHICAL_DIR, category)

            if os.path.isdir(category_w8a8_path):

                # List all files in the category directory
                for file in os.listdir(category_w8a8_path):

                    src_w8a8_file = os.path.join(category_w8a8_path, file)
                    dst_w8a8_file = os.path.join(W8A8_DIR, file)

                    if os.path.exists(dst_w8a8_file):
                        continue

                    shutil.copy2(src_w8a8_file, dst_w8a8_file)

                    src_fp16_file = os.path.join(category_fp16_path, file)
                    dst_fp16_file = os.path.join(FP16_DIR, file)
                    shutil.copy2(src_fp16_file, dst_fp16_file)

                    src_ori_file = os.path.join(category_original_path, file.split(".")[0] + ".jpg")
                    dst_ori_file = os.path.join(MJ_DIR, file)
                    Image.open(src_ori_file).save(dst_ori_file, "PNG")

# from pytorch_fid import fid_score

# fid_value = fid_score.calculate_fid_given_paths([MJ_DIR, FP16_DIR], 
#                                                 batch_size=50, 
#                                                 device='cuda', 
#                                                 dims=2048)
# print('------------------------------------')
# print(f"FID Score (FP16 Unquantized): {fid_value}")
# print('------------------------------------')

# fid_value = fid_score.calculate_fid_given_paths([MJ_DIR, W8A8_DIR], 
#                                                 batch_size=50, 
#                                                 device='cuda', 
#                                                 dims=2048)
# print('------------------------------------')
# print(f"FID Score (W8A8 Quantized): {fid_value}")
# print('------------------------------------')

# Initialize LPIPS loss function
lpips_fn = lpips.LPIPS(net='vgg')  # Options: 'alex', 'vgg', 'squeeze'

def calculate_lpips(true_dir, gen_dir):
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    lpips_values = []

    for true_img_name in os.listdir(true_dir):
        gen_img_name = true_img_name  # Assume matching filenames
        true_img_path = os.path.join(true_dir, true_img_name)
        gen_img_path = os.path.join(gen_dir, gen_img_name)

        true_img = transform(Image.open(true_img_path).convert('RGB')).unsqueeze(0)
        gen_img = transform(Image.open(gen_img_path).convert('RGB')).unsqueeze(0)

        lpips_value = lpips_fn(true_img, gen_img).item()
        lpips_values.append(lpips_value)
    
    return sum(lpips_values) / len(lpips_values)

lpips_score = calculate_lpips(FP16_DIR, W8A8_DIR)
print('------------------------------------')
print(f"Average LPIPS: {lpips_score}")
print('------------------------------------')

import numpy as np

def calculate_psnr(true_dir, gen_dir):
    psnr_values = []

    for true_img_name in os.listdir(true_dir):
        gen_img_name = true_img_name  # Assume matching filenames
        true_img_path = os.path.join(true_dir, true_img_name)
        gen_img_path = os.path.join(gen_dir, gen_img_name)

        true_img = np.array(Image.open(true_img_path).convert('RGB'))
        gen_img = np.array(Image.open(gen_img_path).convert('RGB'))

        print("HI", np.max(true_img))

        mse = np.mean((true_img - gen_img) ** 2)

        # Calculate PSNR
        max_pixel = 255.0  # Assuming 8-bit images
        psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))

        psnr_values.append(psnr_value)

    return sum(psnr_values) / len(psnr_values)

psnr_score = calculate_psnr(FP16_DIR, W8A8_DIR)
print('------------------------------------')
print(f"Average PSNR: {psnr_score} dB")
print('------------------------------------')

from transformers import CLIPProcessor, CLIPModel
import torch
import json

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def calculate_ir(gen_dir, id_to_prompts_dict):

    similarities = []

    for img_name in os.listdir(gen_dir):

        try:

            gen_img_path = os.path.join(gen_dir, img_name)
            img_id = img_name.split(".")[0]
        
            image = Image.open(gen_img_path).convert("RGB")
            prompt = id_to_prompts_dict[img_id]["prompt"]

            inputs = processor(text=prompt, images=image, return_tensors="pt", padding=True)

            # Compute embeddings
            with torch.no_grad():
                outputs = model(**inputs)
                image_emb = outputs.image_embeds
            text_emb = outputs.text_embeds

            # Calculate cosine similarity
            similarity = torch.nn.functional.cosine_similarity(image_emb, text_emb).item()
            similarities.append(similarity)
        
        except:

            print(f"Failed for {img_name}! (Likely due to token sequence length being too high)")
    
    return np.mean(similarities)

with open("/home/user/fq-diffusion/mjhq_dataset/metadata.json", "r") as file:
    full_metadata = json.load(file)

ir_score = calculate_ir(FP16_DIR, full_metadata)

print('------------------------------------')
print(f"IR Score (FP16 Unquantized): {ir_score * 100:.2f}%")
print('------------------------------------')

ir_score = calculate_ir(W8A8_DIR, full_metadata)

print('------------------------------------')
print(f"IR Score (W8A8 Quantized): {ir_score * 100:.2f}%")
print('------------------------------------')

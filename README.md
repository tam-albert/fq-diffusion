# Affine Transformations (FlatQuant) for Diffusion-Based Transformer Models

In this work, we apply **invertible affine transformations** before performing post-training quantization to reduce outliers and quantization error for diffusion models. This approach is inspired by **FlatQuant**, which applies affine transformations for language models. However, we optimize this idea specifically for **transformer-based diffusion models**.


<div align="center">
    <img width="509" alt="image" src="https://github.com/user-attachments/assets/403584d5-80e1-4c0c-a721-05c4aedcb790" />
</div>

---

## Demo

To run the demo (inference) for this project, use the following command:

```bash
python ./main.py \
    --model ./modelzoo/pixart-sigma/PixArt-Sigma-XL-2-1024-MS \
    --w_bits 8 --a_bits 8 \
    --k_bits 8 --k_asym --k_groupsize 128 \
    --v_bits 8 --v_asym --v_groupsize 128 \
    --cali_dataset coco \
    --nsamples 4 --cali_timesteps 10 \
    --cali_bsz 4 --flat_lr 5e-3 \
    --lwc --lac --cali_trans --add_diag \
    --output_dir ./outputs --resume --reload_matrix \
    --prompt "[YOUR PROMPT HERE]"
```

## Parameters:
``w_bits``, ``a_bits``, ``k_bits``, ``v_bits``: Quantization levels for weights, activations, keys, and values.
``--prompt``: Set your text-prompt (default: "A beautiful world").
Recommended Settings: W8A8 or W6A6 (adjust K, V as needed).
The generated image will be saved as ``./demo_image.png``.






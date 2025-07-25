#!/bin/bash
#SBATCH --job-name=indi-sd15
#SBATCH --partition=preemptgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err


# Hugging Face Hub 
HF_MODEL_ID="camgitblame/indiana-sd15"


# Run training
accelerate launch train_dreambooth.py \
    --train_text_encoder \
    --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    --instance_data_dir="indiana/data" \
    --output_dir="indiana/indiana-sd15" \
    --instance_prompt="a photo of idj indianajones" \
    --resolution=512 \
    --mixed_precision="fp16" \
    --train_batch_size=1 \
    --learning_rate=5e-6 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=500 \
    --gradient_accumulation_steps=1 \
    --resume_from_checkpoint="latest" \
    --report_to wandb \
    --push_to_hub \
    --hub_model_id $HF_MODEL_ID



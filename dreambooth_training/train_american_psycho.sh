#!/bin/bash
#SBATCH --job-name=as2
#SBATCH --partition=preemptgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=as_log/%x_%j.out
#SBATCH --error=as_log/%x_%j.err


# Hugging Face Hub 
HF_MODEL_ID="camgitblame/american_psycho_sd15_1500"


# Run training
accelerate launch train_dreambooth.py \
    --train_text_encoder \
    --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    --instance_data_dir="american_psycho/data" \
    --output_dir="american_psycho/american_psycho_sd15" \
    --instance_prompt="a photo of sks americanpsycho" \
    --resolution=512 \
    --mixed_precision="fp16" \
    --train_batch_size=1 \
    --learning_rate=2e-6  \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=1500 \
    --gradient_accumulation_steps=1 \
    --report_to wandb \
    --push_to_hub \
    --hub_model_id $HF_MODEL_ID



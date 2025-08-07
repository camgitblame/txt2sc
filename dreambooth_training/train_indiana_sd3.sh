#!/bin/bash
#SBATCH --job-name=indi-sd3-1000
#SBATCH --partition=preemptgpu
#SBATCH --gres=gpu:a100_80g:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Hugging Face Hub 
HF_MODEL_ID="camgitblame/indiana-sd3"

accelerate launch train_dreambooth_sd3.py \
  --pretrained_model_name_or_path="indiana/indiana-sd3"  \
  --instance_data_dir="indiana/data" \
  --output_dir="indiana/indiana-sd3" \
  --mixed_precision="fp16" \
  --instance_prompt="a photo of idj indianajones" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1500 \
  --resume_from_checkpoint=latest \
  --max_sequence_length=128 \
  --push_to_hub \
  --hub_model_id $HF_MODEL_ID
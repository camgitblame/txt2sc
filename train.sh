#!/bin/bash
#SBATCH --job-name=gray2real_controlnet
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
HF_MODEL_ID="camgitblame/controlnet-fs2k"

# Launch training
accelerate launch \
  --mixed_precision fp16 \
  diffusers/examples/controlnet/train_controlnet.py \
    --pretrained_model_name_or_path ./models/stable-diffusion-v1-5-local \
    --train_data_dir fs2k_split/train \
    --resolution 512 \
    --train_batch_size 4 \
    --num_train_epochs 10 \
    --checkpointing_steps 500 \
    --validation_prompt \
      "a photo of a face" "a photo of a face" "a photo of a face" "a photo of a face" "a photo of a face" \
      "a photo of a face" "a photo of a face" "a photo of a face" "a photo of a face" "a photo of a face" \
    --validation_image \
      fs2k_split/train/sketch/photo1_0001.jpg \
      fs2k_split/train/sketch/photo1_0024.jpg \
      fs2k_split/train/sketch/photo1_0071.jpg \
      fs2k_split/train/sketch/photo1_1664.jpg \
      fs2k_split/train/sketch/photo2_0005.jpg \
      fs2k_split/train/sketch/photo2_0085.jpg \
      fs2k_split/train/sketch/photo2_0097.jpg \
      fs2k_split/train/sketch/photo3_0030.jpg \
      fs2k_split/train/sketch/photo3_0055.jpg \
      fs2k_split/train/sketch/photo3_0102.jpg \
    --num_validation_images 2 \
    --validation_steps 500 \
    --report_to wandb \
    --tracker_project_name controlnet-fs2k \
    --push_to_hub \
    --hub_model_id $HF_MODEL_ID

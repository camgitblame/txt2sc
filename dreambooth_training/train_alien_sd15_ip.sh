#!/bin/bash
#SBATCH --job-name=alien-ip
#SBATCH --partition=preemptgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=alien_log/%x_%j.out
#SBATCH --error=alien_log/%x_%j.err
# ---------------------------
# Optional: W&B tracking via Accelerate
export WANDB_PROJECT="dreambooth-inpaint"
export WANDB_RUN_GROUP="alien_sd15_inpaint"
# ---------------------------
# Hugging Face Hub repo name for the fine-tuned model
HF_MODEL_ID="camgitblame/alien_sd15_inpaint"
# Choose the correct INPAINTING base model
BASE_MODEL="runwayml/stable-diffusion-inpainting"   # SD 1.5 inpaint, 512px
# Paths
INSTANCE_DIR="alien/data"
OUTPUT_DIR="alien/alien_sd15_inpaint"

# Run training (research_projects/dreambooth_inpaint)
accelerate launch train_dreambooth_inpaint.py \
 --train_text_encoder \
  --pretrained_model_name_or_path="$BASE_MODEL" \
  --instance_data_dir="$INSTANCE_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --instance_prompt="a photo of sks alien" \
  --resolution=512 \
  --mixed_precision="fp16" \
  --train_batch_size=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --gradient_accumulation_steps=1 \
  --use_8bit_adam \
  --gradient_checkpointing \
  --push_to_hub \
  --hub_model_id "$HF_MODEL_ID"

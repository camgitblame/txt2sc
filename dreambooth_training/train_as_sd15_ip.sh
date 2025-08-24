#!/bin/bash
#SBATCH --job-name=psy15-inpaint
#SBATCH --partition=preemptgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=as_log/%x_%j.out
#SBATCH --error=as_log/%x_%j.err

# ---------------------------
# Optional: W&B tracking via Accelerate
export WANDB_PROJECT="dreambooth-inpaint"
export WANDB_RUN_GROUP="american_psycho_sd15_inpaint"
# ---------------------------

# Hugging Face Hub repo name for the fine-tuned model
HF_MODEL_ID="camgitblame/american_psycho_sd15_inpaint"

# Choose the correct INPAINTING base model
BASE_MODEL="runwayml/stable-diffusion-inpainting"   # SD 1.5 inpaint, 512px
# For SD 2.x inpainting use:
# BASE_MODEL="stabilityai/stable-diffusion-2-inpainting"  # then set --resolution=768

# Paths
INSTANCE_DIR="american_psycho/data"                 
OUTPUT_DIR="american_psycho/american_psycho_sd15_inpaint"

# Run training (research_projects/dreambooth_inpaint)
accelerate launch train_dreambooth_inpaint.py \
 --train_text_encoder \
  --pretrained_model_name_or_path="$BASE_MODEL" \
  --instance_data_dir="$INSTANCE_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --instance_prompt="a photo of sks americanpsycho" \
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
  --report_to wandb \
  --hub_model_id "$HF_MODEL_ID"

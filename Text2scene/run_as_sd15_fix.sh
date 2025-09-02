#!/bin/bash
#SBATCH --job-name=asfix
#SBATCH --partition=preemptgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=as_log/%x_%j.out
#SBATCH --error=as_log/%x_%j.err

export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME"/{hub,transformers,datasets}

python run_fix.py --base-config ./config/base-config.yaml --example_config ./config/example_configs/as_sd15_30frames_fix.yaml
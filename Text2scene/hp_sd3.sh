#!/bin/bash
#SBATCH --job-name=hp5
#SBATCH --partition=preemptgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Set environment variable to suppress tokenizers warning
export TOKENIZERS_PARALLELISM=false

python run.py --base-config ./config/base-config_sd3.yaml --example_config ./config/example_configs/hp_sd3.yaml
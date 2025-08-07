#!/bin/bash
#SBATCH --job-name=hp2
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

python run.py --base-config ./config/base-config.yaml --example_config ./config/example_configs/hp_sd15.yaml
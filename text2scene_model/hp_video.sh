#!/bin/bash
#SBATCH --job-name=hp-vid
#SBATCH --partition=preemptgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

python run.py --base-config ./config/base-config.yaml --example_config ./config/example_configs/hp.yaml
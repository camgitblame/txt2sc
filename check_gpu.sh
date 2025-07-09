#!/bin/bash
#SBATCH --job-name=gpu-mem
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=logs/gpu_mem_%j.out

module load cuda  # Only if required on your cluster

python -c "import torch; print(f'Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')"

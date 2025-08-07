#!/bin/bash
# === ENV SETUP ===
source /opt/flight/etc/setup.sh
flight env activate gridware
module load gnu
module load compilers/gcc
module load libs/nvidia-cuda/12.2.2/bin  

# === PROXY ===
export https_proxy=http://hpc-proxy00.city.ac.uk:3128
export http_proxy=http://hpc-proxy00.city.ac.uk:3128
export TORCH_HOME=/mnt/data/public/torch

# === VENV SETUP ===
which python
python --version
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv virtualenv-init -)"

pyenv virtualenv 3.9 txt2sc
source ~/envs/txt2sc/bin/activate

# === UPGRADE PIP ===
pip install --upgrade pip setuptools wheel

python --version

# === INSTALL REQUIREMENTS ===
pip install --proxy $https_proxy -r requirements_text2sc


# === INSTALL TORCH (CUDA) ===
pip install --proxy $https_proxy torch torchvision --index-url https://download.pytorch.org/whl/cu121

# === CHECK PYTHON ENV ===
echo "Python and pip versions:"
which python
which pip
pip list --format=columns

# === Login to wandb ===
export WANDB_API_KEY= wandb_key
wandb login $WANDB_API_KEY --relogin

# === Login to Hugging Hace ===
huggingface-cli login --token hf_token

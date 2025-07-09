#!/bin/bash
# === ENV SETUP ===
source /opt/flight/etc/setup.sh
flight env activate gridware
module load gnu
module load compilers/gcc
module load libs/nvidia-cuda/11.2.0/bin  

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

source ~/envs/txt2sc/bin/activate


# === UPGRADE PIP ===
pip install --upgrade pip setuptools wheel

python --version

# === INSTALL REQUIREMENTS ===
if [ -f requirements.txt ]; then
    pip install --proxy $https_proxy -r requirements_dreambooth.txt
fi

# === INSTALL TORCH (CUDA) ===
pip install --proxy $https_proxy torch torchvision --index-url https://download.pytorch.org/whl/cu118

# === CHECK PYTHON ENV ===
echo "Python and pip versions:"
which python
which pip
pip list --format=columns

# === Login to wandb ===
export WANDB_API_KEY=f1b1dcb5ebf893b856630d4481b7d7cd05101b45
wandb login $WANDB_API_KEY --relogin
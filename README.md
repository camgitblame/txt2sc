# Text2Scene

Stylized, depth-consistent 3D scene generation for movies from text prompts, using DreamBooth, Stable Diffusion 3 and ControlNet.


## Setup

- Clone the repository and navigate to the project directory.

- Run `setup.sh` to set up your environment and required modules, install all dependencies and configure network proxies for cluster access.

- Update `wandb_key` and `hf_token` in the script with your credentials to log into wandb and HuggingFace.


```bash
source setup.sh

```

## Training

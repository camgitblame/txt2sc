# Text2Scene

## Overview

This project generates cinematic 3D scene walkthroughs from text descriptions by combining Stable Diffusion 3 with DreamBooth training. Built upon the SceneScape framework, it creates depth-consistent movie environments with enhanced visual quality and geometric consistency across frames.


### Project Structure

```
├── Text2scene/                    # Main generation pipeline with SD3 integration
├── SceneScape_baseline/           # Original SceneScape implementation
├── dreambooth_training/           # DreamBooth training scripts 
├── generation_sampler/            # Standalone generation scripts
├── data/                          # Training datasets and movie scene references
├── output/                        # Generated videos and intermediate results
├── setup/                         # Environment config and dependencies
└── sd3/                          # Stable Diffusion 3 model files
```


## Setup

### Installation

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/camgitblame/txt2sc.git
cd txt2sc
```

Run the setup script to configure environment

```bash
source setup/setup.sh
```

**Note**: Update `wandb_key` and `hf_token` in the setup script to log into wandb and HuggingFace.

### Environment Setup

```bash
source setup/activate_env.sh
```

## Usage

Generate a scene walkthrough video:

```bash
cd Text2scene
python run.py --config config/example_configs/hp.yaml
```


## Training

### DreamBooth 

```bash
cd dreambooth_training

# Train on sceneswith the Great Hall from Harry Potter
bash train_hp_sd15.sh
```

## Related Work

- [SceneScape](https://arxiv.org/abs/2302.01133) - Original text-to-3D scene generation
- [DreamBooth](https://arxiv.org/abs/2208.12242) - Few-shot style adaptation
- [Stable Diffusion 3](https://stability.ai/news/stable-diffusion-3) - Advanced text-to-image generation


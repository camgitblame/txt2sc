# Text2Scene

## Overview

This project generates 3D scene walkthroughs for movies from text by combining Stable Diffusion with DreamBooth and multi-ControlNet setup. Built upon the SceneScape framework, it creates depth-consistent movie environments with enhanced visual quality and geometric consistency across frames.


### Project Structure

```
├── Text2scene/                    # Main pipeline with DreamBooth and ControlNet integration
├── SceneScape_baseline/           # Baseline (SceneScape)
├── dreambooth_training/           # DreamBooth training scripts 
├── generation_sampler/            # Generation scripts to test DreamBooth checkpoints
├── notebook/                      # Notebooks for computing quatitative metrics, visualizing data, sample frames and output video frames 
├── setup/                         # Environment config and dependencies

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
python run.py --config config/example_configs/the_shining.yaml
```


## Training

### DreamBooth 

```bash
cd dreambooth_training

# Train on images from the Overlook Hotel Hallway from The Shining (1980)
bash train_the_shining.sh
```

## Related Work

- [SceneScape](https://arxiv.org/abs/2302.01133) - Original text-to-3D scene generation
- [DreamBooth](https://arxiv.org/abs/2208.12242) - Few-shot style adaptation



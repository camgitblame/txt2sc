# Text2Scene

## Overview

This project presents a text-to-3D pipeline that generates aesthetically pleasing, perpetual scene walkthroughs aligned with the visual identity of target films. Building on the SceneScape baseline, we fine-tune Stable Diffusion with DreamBooth for few-shot, scene-focused synthesis that recreates each film’s color palette, materials, and set dressing.To maintain stable geometry under camera motion, we guide inpainting with a multi-ControlNet setup that conditions masks using ControlNet-Inpaint and ControlNet-Depth. At test time, we add four lightweight stabilizers, namely EMA-smoothing for depth, seam-aware mask morphology, immediate mesh accumulation, and a short camera-motion warm-up, which improves structural consistency over long video sequences.We evaluate the results on five stylistically distinct movies. In qualitative analysis, both human experts and GPT-4V strongly prefer our outputs over the baseline for film likeness, visual quality, 3D structural consistency, and prompt alignment. Quantitatively, CLIP-AS and reconstructed 3D density increase over the baseline, indicating more appealing frames and fuller coverage, while reprojection error and CLIP-TS remain comparable to SceneScape. Overall, our results improve on the baseline and provide a practical path to film-specific, 3D-plausible walkthroughs that require no 3D or multiview training data.

### Project Links and Resources

- [Project Page](https://text2scene.vercel.app/) 
The output videos of all models are available in our project pages. 

- [DreamBooth Checkpoints](https://huggingface.co/camgitblame) 


### Project Structure

```
├── Text2scene/                              # Main pipeline with DreamBooth and ControlNet integration
│   ├── config/                              # Config files to run DB+CN models for each scene
│   │   ├── base-config.yaml                 # Base configuration file
│   │   └── example_configs/                 # Config files for different scenes
│   │       ├── thesubstance_DB1CN_config.yaml   # Example config file for DB+1CN 
│   │       └── thesubstance_DB2CN_config.yaml   # Example config file for DB+2CN
│   ├── models/                              # Implementation files
│   │   ├── warp_inpaint_model_controlnet.py # WarpInpaint Model - Scene Generation Script
│   │   └── warp_fix.py                      # DepthControlNet and Stabilizers Integration Script       
│   ├── run_text2scene.py                    # Main execution script for Text2Scene pipeline
│   └── run_thesubstance_DB1CN.sh            # Example shell script to run pipeline and generate scene 
│                                            # with the model DB+1CN for The Substance
├── SceneScape_baseline/                     # Baseline (SceneScape)
│   ├── config/                              # Config files to run baseline model for each scene
│   ├── models/                              # Baseline model implementations 
│   └── run.py                               # Baseline execution script
├── dreambooth_training/                     # DreamBooth training scripts 
├── generation_sampler/                      # Generation scripts to test DreamBooth checkpoints
├── notebook/                                # Notebooks for computing quantitative metrics, visualizing data, 
│                                            # sample frames and output video frames 
└── setup/                                   # Environment config and dependencies
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

Generate a movie walkthrough video of the Overlook Hotel hallway from The Shining (1980) using the DB+1CN model:

```bash
cd Text2scene
python run_text2scene.py --base-config ./config/base-config.yaml --example_config ./config/example_configs/thesubstance_DB1CN_config.yaml
```
or 

Use the shell script:
```bash
cd Text2scene
bash run_theshining_DB1CN.sh
```


## Training DreamBooth

### DreamBooth 

```bash
cd dreambooth_training

# Train on images from the Overlook Hotel Hallway from The Shining (1980)
bash train_the_shining.sh
```



## Related Work

- [SceneScape](https://arxiv.org/abs/2302.01133) - Original text-to-3D scene generation
- [DreamBooth](https://arxiv.org/abs/2208.12242) - Few-shot style adaptation



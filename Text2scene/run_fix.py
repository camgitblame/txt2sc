import copy
import gc
import random
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.cuda.amp import GradScaler
from torchvision.transforms import ToPILImage
from tqdm import tqdm

# === Import the enhanced ControlNet model ===
from models.warp_inpaint_model_fix import WarpInpaintModel
from util.finetune_utils import finetune_depth_model, finetune_decoder
from util.general_utils import apply_depth_colormap, save_video


def evaluate(model):
    fps = model.config["save_fps"]
    save_root = Path(model.run_dir)
    save_dict = {
        "images": torch.cat(model.images, dim=0),
        "images_orig_decoder": torch.cat(model.images_orig_decoder, dim=0),
        "masks": torch.cat(model.masks, dim=0),
        "disparities": torch.cat(model.disparities, dim=0),
        "depths": torch.cat(model.depths, dim=0),
        "cameras": model.cameras_extrinsics,
    }
    torch.save(save_dict, save_root / "results.pt")
    if not model.config["use_splatting"]:
        model.save_mesh("full_mesh")

    video = (255 * torch.cat(model.images, dim=0)).to(torch.uint8).detach().cpu()
    video_reverse = (
        (255 * torch.cat(model.images[::-1], dim=0)).to(torch.uint8).detach().cpu()
    )

    save_video(video, save_root / "output.mp4", fps=fps)
    save_video(video_reverse, save_root / "output_reverse.mp4", fps=fps)


def evaluate_epoch(model, epoch):
    disparity = model.disparities[epoch]
    disparity_colored = apply_depth_colormap(disparity[0].permute(1, 2, 0))
    disparity_colored = disparity_colored.clone().permute(2, 0, 1).unsqueeze(0).float()
    save_root = Path(model.run_dir) / "images"
    save_root.mkdir(exist_ok=True, parents=True)
    (save_root / "frames").mkdir(exist_ok=True, parents=True)
    (save_root / "images_orig_decoder").mkdir(exist_ok=True, parents=True)
    (save_root / "masks").mkdir(exist_ok=True, parents=True)
    (save_root / "warped_images").mkdir(exist_ok=True, parents=True)
    (save_root / "disparities").mkdir(exist_ok=True, parents=True)

    ToPILImage()(model.images[epoch][0]).save(save_root / "frames" / f"{epoch}.png")
    ToPILImage()(model.images_orig_decoder[epoch][0]).save(
        save_root / "images_orig_decoder" / f"{epoch}.png"
    )
    ToPILImage()(model.masks[epoch][0]).save(save_root / "masks" / f"{epoch}.png")
    ToPILImage()(model.warped_images[epoch][0]).save(
        save_root / "warped_images" / f"{epoch}.png"
    )
    ToPILImage()(disparity_colored[0]).save(save_root / "disparities" / f"{epoch}.png")

    if epoch == 0:
        with open(Path(model.run_dir) / "config.yaml", "w") as f:
            OmegaConf.save(model.config, f)


def run(config):
    # === Enhanced setup with DreamBooth + ControlNet validation ===
    seed = config["seed"]
    if seed == -1:
        seed = np.random.randint(2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"Running with seed: {seed}")

    # Validate configuration for DreamBooth + ControlNet setup
    print("\n=== CONFIGURATION VALIDATION ===")
    print(f"Model checkpoint: {config['stable_diffusion_checkpoint']}")
    print(f"Using DreamBooth UNet: {config.get('use_dreambooth_unet', False)}")
    print(f"Using ControlNet: {config.get('use_controlnet', False)}")

    # Enhanced geometry fixes validation
    warp_fix_enabled = config.get("enable_warp_fix", True)
    print(f"Geometry fixes enabled: {warp_fix_enabled}")

    if warp_fix_enabled:
        fix_overrides = config.get("warp_fix_overrides", {})
        print(f"Geometry fix overrides: {fix_overrides}")

        # Validate fix configuration
        fix_strength = fix_overrides.get("fix_inpaint_strength", 0.7)
        fix_cn_scale = fix_overrides.get("fix_depth_cn_scale", 0.8)
        print(f"Fix inpaint strength: {fix_strength}")
        print(f"Fix depth ControlNet scale: {fix_cn_scale}")

        if fix_strength < 0.3 or fix_strength > 1.0:
            print("WARNING: fix_inpaint_strength should be between 0.3-1.0")
        if fix_cn_scale < 0.5 or fix_cn_scale > 1.5:
            print("WARNING: fix_depth_cn_scale should be between 0.5-1.5")

    if config.get("use_dreambooth_unet", False):
        print("Hybrid approach enabled: Base SD1.5 inpainting + DreamBooth UNet")
    else:
        print("Direct approach: Using DreamBooth inpainting model")

    if config.get("use_controlnet", False):
        controlnet_model = config.get(
            "controlnet_model", "lllyasviel/control_v11p_sd15_inpaint"
        )
        controlnet_scale = config.get("controlnet_conditioning_scale", 1.0)
        print(f"ControlNet enabled: {controlnet_model} (scale: {controlnet_scale})")

        # Validate ControlNet scale
        if controlnet_scale < 0.5 or controlnet_scale > 1.5:
            print("WARNING: controlnet_conditioning_scale should be between 0.5-1.5")
    else:
        print("ControlNet disabled")

    print(f"Frames to generate: {config['frames']}")
    print(f"Inpainting steps: {config['num_inpainting_steps']}")
    print("=== STARTING GENERATION ===\n")

    # Initialize model with enhanced error handling
    try:
        model = WarpInpaintModel(config).to(config["device"])
        print("Model initialization successful")
    except Exception as e:
        print(f"Model initialization failed: {e}")
        print("Check model path and configuration flags")
        raise
    evaluate_epoch(model, 0)
    scaler = GradScaler(enabled=config["enable_mix_precision"])

    # === Enhanced generation loop with progress tracking ===
    for epoch in tqdm(range(1, config["frames"] + 1), desc="Generating frames"):
        print(f"\n--- Frame {epoch}/{config['frames']} ---")

        # [ADD 1] motion schedule tweak (safe even if patch isn't loaded)
        if hasattr(model, "apply_motion_schedule"):
            model.apply_motion_schedule(epoch, config["frames"])

        try:
            # Warping step
            print("Warping...")
            if config["use_splatting"]:
                warp_output = model.warp_splatting(epoch)
            else:
                warp_output = model.warp_mesh(epoch)
            print("Warping completed")

            # Inpainting step (with DreamBooth + ControlNet)
            print("Inpainting...")
            inpaint_output = model.inpaint(
                warp_output["warped_image"], warp_output["inpaint_mask"]
            )
            print("Inpainting completed")

            # Optional decoder finetuning
            if config["finetune_decoder"]:
                print("Finetuning decoder...")
                finetune_decoder(config, model, warp_output, inpaint_output)
                print("Decoder finetuning completed")

            model.update_images_masks(
                inpaint_output["latent"], warp_output["inpaint_mask"]
            )

            # [ADD 2] geometry accumulation helper (uses just-appended frame/mask)
            if hasattr(model, "accumulate_mesh_after_inpaint"):
                model.accumulate_mesh_after_inpaint(
                    model.images[epoch],  # already cropped to 512 if needed
                    model.masks[epoch],
                    epoch,
                )

            # Optional depth model finetuning
            if config["finetune_depth_model"]:
                print("Finetuning depth model...")
                # reload depth model
                del model.depth_model
                gc.collect()
                torch.cuda.empty_cache()
                model.depth_model = torch.hub.load("intel-isl/MiDaS", "DPT_Large").to(
                    model.device
                )

                finetune_depth_model(config, model, warp_output, epoch, scaler)
                print("Depth model finetuning completed")

            model.update_depth(model.images[epoch])

            if not config["use_splatting"]:
                print("Updating mesh...")
                # update mesh with the correct mask
                if config["mask_opening_kernel_size"] > 0:
                    mesh_mask = 1 - torch.maximum(
                        model.masks[epoch], model.masks_diffs[epoch - 1]
                    )
                else:
                    mesh_mask = 1 - model.masks[epoch]
                extrinsic = model.get_extrinsics(model.current_camera)
                model.update_mesh(
                    model.images[epoch],
                    model.depths[epoch],
                    mesh_mask > 0.5,
                    extrinsic,
                    epoch,
                )
                print("Mesh updated")

            # reload decoder
            model.vae.decoder = copy.deepcopy(model.decoder_copy)

            model.images_orig_decoder.append(
                model.decode_latents(inpaint_output["latent"]).detach()
            )
            evaluate_epoch(model, epoch)

            print(f"Frame {epoch} completed successfully")

            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"Error in frame {epoch}: {e}")
            print("Check your configuration and model compatibility")
            raise

    print("Generating final video...")
    evaluate(model)
    print("Generation completed successfully!")
    print(f"Output saved to: {model.run_dir}")


if __name__ == "__main__":
    # === Enhanced argument parser with better defaults ===
    parser = ArgumentParser(description="Run DreamBooth + ControlNet scene generation")
    parser.add_argument(
        "--base-config",
        default="./config/base-config.yaml",
        help="Path to base configuration file",
    )
    parser.add_argument(
        "--example_config",
        default="./config/example_configs/as_sd15.yaml",  # Updated default to your config
        help="Path to example configuration file (for DreamBooth + ControlNet)",
    )
    args = parser.parse_args()

    print("=== LOADING CONFIGURATION ===")
    print(f"Base config: {args.base_config}")
    print(f"Example config: {args.example_config}")

    try:
        base_config = OmegaConf.load(args.base_config)
        example_config = OmegaConf.load(args.example_config)
        config = OmegaConf.merge(base_config, example_config)
        print("Configuration loaded successfully")
    except Exception as e:
        print(f"Configuration loading failed: {e}")
        print("Check config file paths")
        raise

    run(config)

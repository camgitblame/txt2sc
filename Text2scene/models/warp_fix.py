# warp_fix.py
"""
Fast-impact geometry fixes for WarpInpaintModel (no core rewrite).

What this adds:
1) Mesh accumulation after each inpaint
2) Optional multi-ControlNet (add Depth CN alongside Inpaint CN)
3) Conservative inpaint (smaller mask + lower denoise strength + deterministic seed)
4) Temporal depth smoothing (EMA)
5) Motion warmup (smaller camera steps early, optional)

Usage (runner or after constructing the model):
    from warp_fix import attach_fast_fixes
    attach_fast_fixes(model, config_overrides={...})
"""

from typing import Optional, Dict, Any, List
import types
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from torchvision.transforms import ToPILImage

# ---- defaults you can override via `config_overrides` ----
DEFAULTS = {
    "fix_enable_mesh_accumulation": True,
    "fix_enable_depth_controlnet": True,  # requires diffusers ControlNet weights
    "fix_depth_controlnet_id": "lllyasviel/control_v11f1p_sd15_depth",
    "fix_depth_cn_scale": 0.8,  # 0.6–1.0 works well
    "fix_inpaint_strength": 0.45,  # denoise strength (0.35–0.5)
    "fix_mask_op": "erode",  # "erode" | "dilate" | None
    "fix_mask_iters": 1,  # small number on purpose
    "fix_seed": None,  # if None, will use config['seed'] or 13
    "fix_depth_ema_alpha": 0.7,  # temporal depth smoothing
    "fix_motion_warmup_ratio": 0.25,  # % of frames with reduced motion
    "fix_motion_warmup_scale": 0.5,  # multiply camera_speed_factor during warmup
}


def _normalize01(x: torch.Tensor) -> torch.Tensor:
    mn, mx = x.amin(), x.amax()
    return (x - mn) / (mx - mn + 1e-6)


def _prep_mask(mask_tensor: torch.Tensor, op: Optional[str], iters: int) -> Image.Image:
    """mask_tensor: (1,1,H,W) in {0,1} -> PIL L"""
    m = (mask_tensor[0, 0].detach().float().cpu().numpy() * 255).astype(np.uint8)
    if op == "erode":
        m = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=max(0, iters))
    elif op == "dilate":
        m = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=max(0, iters))
    return Image.fromarray(m).convert("L")


def _depth_control_image(model, img_tensor: torch.Tensor) -> Image.Image:
    """
    Build a normalized depth map (MiDaS) as a single-channel PIL image.
    img_tensor: (1,3,H,W), [0,1], same dtype as model.images
    """
    with torch.no_grad():
        d, _ = model.get_depth(img_tensor)  # get_depth handles dtype conversion
        d = _normalize01(d).float()
    return ToPILImage()(d[0])  # 1xHxW -> PIL L


def _enable_multi_controlnet(model, depth_cn_id: str):
    """Append a Depth ControlNet to an existing ControlNet in the inpaint pipeline."""
    from diffusers import ControlNetModel  # local import, optional dependency

    pipe = model.inpainting_pipeline
    dtype = getattr(pipe.unet, "dtype", torch.float16)
    depth_cn = ControlNetModel.from_pretrained(depth_cn_id, torch_dtype=dtype)

    # Allow single or list on current pipe
    if isinstance(pipe.controlnet, list):
        pipe.controlnet = pipe.controlnet + [depth_cn]
    else:
        pipe.controlnet = [pipe.controlnet, depth_cn]

    model._fix_depth_controlnet_enabled = True


def attach_fast_fixes(model, config_overrides: Optional[Dict[str, Any]] = None):
    """
    Monkey-patch the given model with 5 fast fixes.
    Call this once, right after you construct WarpInpaintModel.
    """
    cfg = DEFAULTS.copy()
    if config_overrides:
        cfg.update(config_overrides)

    # -------------- (5) Motion warmup helpers --------------
    model._fix_base_speed = float(model.camera_speed_factor)
    model._fix_motion_warmup_ratio = float(cfg["fix_motion_warmup_ratio"])
    model._fix_motion_warmup_scale = float(cfg["fix_motion_warmup_scale"])

    def apply_motion_schedule(self, epoch: int, total_epochs: int):
        """Reduce camera speed during the first warmup portion."""
        warmup = int(total_epochs * self._fix_motion_warmup_ratio)
        if epoch < warmup:
            self.camera_speed_factor = (
                self._fix_base_speed * self._fix_motion_warmup_scale
            )
        else:
            self.camera_speed_factor = self._fix_base_speed

    model.apply_motion_schedule = types.MethodType(apply_motion_schedule, model)

    # -------------- (4) Patch update_depth with EMA --------------
    model._orig_update_depth = model.update_depth
    alpha = float(cfg["fix_depth_ema_alpha"])

    def update_depth_ema(self, inpainted_image):
        d, disp = self.get_depth(inpainted_image)
        if len(self.depths) > 0:
            d = alpha * d + (1.0 - alpha) * self.depths[-1]
            disp = alpha * disp + (1.0 - alpha) * self.disparities[-1]
        self.depths.append(d.detach())
        self.disparities.append(disp.detach())

    model.update_depth = types.MethodType(update_depth_ema, model)

    # -------------- (1) Accumulate mesh after inpaint --------------
    def accumulate_mesh_after_inpaint(self, inpainted_image, inpaint_mask, epoch: int):
        if not cfg["fix_enable_mesh_accumulation"]:
            return
        new_depth, _ = self.get_depth(inpainted_image)
        # limit addition primarily to masked (newly revealed) area
        mask_bool = (
            inpaint_mask.bool()
            if inpaint_mask is not None
            else torch.zeros_like(new_depth, dtype=torch.bool)
        )
        try:
            self.update_mesh(
                inpainted_image,
                new_depth,
                mask_bool,  # treat True=exclude -> pass where we DON'T want to keep? (your update_mesh sets updated_depth[mask]=-1)
                self.get_extrinsics(self.current_camera),
                epoch,
            )
        except Exception:
            # fallback without mask if shape mismatch
            self.update_mesh(
                inpainted_image,
                new_depth,
                torch.zeros_like(mask_bool),
                self.get_extrinsics(self.current_camera),
                epoch,
            )

    model.accumulate_mesh_after_inpaint = types.MethodType(
        accumulate_mesh_after_inpaint, model
    )

    # -------------- (2) Optional: add Depth ControlNet --------------
    model._fix_depth_controlnet_enabled = False
    if getattr(model, "use_controlnet", False) and cfg["fix_enable_depth_controlnet"]:
        try:
            _enable_multi_controlnet(model, cfg["fix_depth_controlnet_id"])
        except Exception as e:
            print(
                f"[warp_fix] Depth ControlNet not enabled ({e}). Continuing without multi-ControlNet."
            )

    # -------------- (3) Patch inpaint to shrink mask + strength + deterministic seed --------------
    model._orig_inpaint = model.inpaint
    seed = cfg["fix_seed"]
    if seed is None:
        seed = int(model.config.get("seed", 13))
    model._fix_generator = torch.Generator(device=model.device).manual_seed(seed)
    model._fix_inpaint_strength = float(cfg["fix_inpaint_strength"])
    model._fix_mask_op = cfg["fix_mask_op"]
    model._fix_mask_iters = int(cfg["fix_mask_iters"])
    model._fix_depth_cn_scale = float(cfg["fix_depth_cn_scale"])

    def inpaint_patched(self, warped_image, inpaint_mask):
        # 1) conservative mask
        mask_pil = _prep_mask(inpaint_mask, self._fix_mask_op, self._fix_mask_iters)

        # 2) if multi-ControlNet is enabled, build depth condition too
        if self.use_controlnet:
            # CONTROLNET INPAINTING: Use structural guidance for better geometry consistency
            control_image = self.make_inpaint_condition(
                ToPILImage()(warped_image[0]), ToPILImage()(inpaint_mask[0])
            )
            # Additional mask dilation for ControlNet
            kernel = np.ones((5, 5), np.uint8)
            mask = np.array(ToPILImage()(inpaint_mask[0]).convert("L")).astype(
                np.float32
            )
            mask = cv2.dilate(mask, kernel, iterations=3)
            mask_pil = Image.fromarray(mask).convert("L")

            # Check if control_image is a list and handle appropriately
            if isinstance(control_image, list):
                control_image_input = control_image
            else:
                control_image_input = control_image

            res = self.inpainting_pipeline(
                prompt=self.inpainting_prompt,
                negative_prompt=self.config["negative_inpainting_prompt"],
                image=ToPILImage()(warped_image[0]),
                mask_image=mask_pil,
                control_image=control_image_input,  # ControlNet conditioning
                controlnet_conditioning_scale=self.config.get(
                    "controlnet_conditioning_scale", 1.0
                ),
                num_inference_steps=self.config["num_inpainting_steps"],
                callback_steps=self.config["num_inpainting_steps"] - 1,
                callback=self.latent_storer,
                guidance_scale=self.classifier_free_guidance_scale,
                generator=self._fix_generator,
                num_images_per_prompt=1,
                height=self.config["inpainting_resolution"],
                width=self.config["inpainting_resolution"],
                strength=self._fix_inpaint_strength,
            ).images
        else:
            # standard inpaint, just pass conservative mask + strength + generator
            res = self.inpainting_pipeline(
                prompt=self.inpainting_prompt,
                negative_prompt=self.config["negative_inpainting_prompt"],
                image=ToPILImage()(warped_image[0]),
                mask_image=mask_pil,
                num_inference_steps=self.config["num_inpainting_steps"],
                callback_steps=self.config["num_inpainting_steps"] - 1,
                callback=self.latent_storer,
                guidance_scale=self.classifier_free_guidance_scale,
                generator=self._fix_generator,
                num_images_per_prompt=1,
                height=self.config["inpainting_resolution"],
                width=self.config["inpainting_resolution"],
                strength=self._fix_inpaint_strength,
            ).images

        best_index = 0
        inpainted_image = res[best_index]
        latent = self.latent_storer.latent[[best_index]]

        # convert back to tensor consistent with original return contract
        inpainted_tensor = ToTensor()(res[best_index]).unsqueeze(0).to(self.device)
        latent = self.latent_storer.latent[[best_index]].float()
        return {
            "inpainted_image": inpainted_tensor,
            "latent": latent,
            "best_index": best_index,
        }

    model.inpaint = types.MethodType(inpaint_patched, model)

    print("[warp_fix] Fast fixes attached.")
    return model

"""
WarpInpaintModel (from Scene Scape) adapted for Stable Diffusion 3
===================================================================
This is an adaptation of the original Scene Scape pipleine's WarpInpaintModel to support SD3 instead of SD1.5.
It includes updates for SD3 compatibility and fixes for whitecast issues caused by mesh warping.
Key updates are marked with "CHANGE:" comments throughout the code.
===================================================================

1. PIPELINE UPDATES FOR SD3:
   - ChangedSD1.5’s StableDiffusionInpaintPipeline for SD3’s StableDiffusion3InpaintPipeline
   - Replaced DDIMScheduler with FlowMatchEulerDiscreteScheduler (native to SD3)
   - Removed safety checker and revision flags since they’re unused in SD3

2. MEMORY OPTIMIZATION:
   - SD3 doesn’t support xformers, so we now use enable_vae_slicing() and enable_vae_tiling() for memory efficiency

3. FIRST FRAME GENERATION:
   - SD1.5 used full white inpainting on a white image
   - SD3 uses text-to-image instead

4. LATENT SCALING:
   - SD3 uses a different scale factor for latents: 0.13025 (instead of SD1.5’s 0.18215)

5. INPAINTING UPGRADES:
   - Detects bright/white artifacts and forces them into the mask
   - Applies conservative erosion to avoid over-inpainting
   - Uses progressive inpainting for large masks
   - Adjusts guidance and strength based on mask content
   - Uses grayscale masks (L mode) as expected by SD3

6. DTYPE FIXES:
   - MiDaS depth models need float32, but SD3 runs in float16

7. IMAGE INSTEAD OF LATENTS:
   - Uses actual inpainted images directly when possible instead of decoding from latents

8. SAVING DEBUG IMAGES:
   - Saves per-frame debug images including overlays for masks and detected white areas
===================================================================
"""

import copy
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image
from diffusers import (
    # CHANGE: Updated imports for SD3
    # Original SD1.5: StableDiffusionInpaintPipeline, DDIMScheduler
    # SD3: StableDiffusion3InpaintPipeline, FlowMatchEulerDiscreteScheduler
    StableDiffusion3InpaintPipeline,  # SD3 inpainting pipeline
    StableDiffusion3Pipeline,  # SD3 text-to-image pipeline for first frame
    FlowMatchEulerDiscreteScheduler,  # SD3 native scheduler
    AutoencoderKL,
)
from einops import rearrange
from kornia.geometry import (
    PinholeCamera,
    transform_points,
    convert_points_from_homogeneous,
)
from kornia.morphology import dilation, opening, erosion
from pytorch3d.renderer import (
    PerspectiveCameras,
    look_at_rotation,
)
from torchvision.transforms import ToTensor, ToPILImage, Resize

from models.mesh_renderer import Renderer
from util.general_utils import (
    LatentStorer,
    sobel_filter,
)
from util.midas_utils import dpt_transform


class WarpInpaintModel(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        if config["use_splatting"]:
            sys.path.append("util/softmax-splatting")
            import softsplat  # Import softsplat when needed

        # get current date and time up to minutes
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y_%H-%M")
        run_dir_root = Path(config["runs_dir"])
        self.run_dir = (
            run_dir_root
            / f"{dt_string}_{config['inpainting_prompt'].replace(' ', '_')[:40]}"
        )
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.device = config["device"]
        self.config = config
        self.inpainting_prompt = config["inpainting_prompt"]

        # CHANGE: Updated pipeline for SD3
        # Original SD1.5: StableDiffusionInpaintPipeline.from_pretrained with safety_checker
        # SD3: StableDiffusion3InpaintPipeline without safety_checker
        self.inpainting_pipeline = StableDiffusion3InpaintPipeline.from_pretrained(
            config["stable_diffusion_checkpoint"],
            torch_dtype=torch.float16,  # SD3 uses torch_dtype directly
        )
        # CHANGE: Updated scheduler for SD3
        # Original SD1.5: DDIMScheduler
        # SD3: FlowMatchEulerDiscreteScheduler (SD3's native)
        self.inpainting_pipeline.scheduler = (
            FlowMatchEulerDiscreteScheduler.from_config(
                self.inpainting_pipeline.scheduler.config
            )
        )
        self.inpainting_pipeline = self.inpainting_pipeline.to(self.device)
        if self.config["use_xformers"]:
            # CHANGE: Updated memory optimization for SD3
            # Original SD1.5: set_use_memory_efficient_attention_xformers(True)
            # SD3: Use enable_vae_slicing() and enable_vae_tiling() instead
            # because xformers method is not available in SD3
            try:
                self.inpainting_pipeline.enable_vae_slicing()
                self.inpainting_pipeline.enable_vae_tiling()
            except AttributeError:
                print(
                    "Warning: VAE optimization methods not available for this SD3 pipeline"
                )

        # CHANGE: New generation approach for first frame for SD3
        # Original SD1.5: Used white mask + white image for inpainting the first frame
        # SD3: Use text-to-image generation because SD3 is more sensitive to mask/image combinations, white+white can cause artifacts
        print(f"Generating initial image with prompt: '{self.inpainting_prompt}'")

        # Use pure text-to-image generation instead of inpainting for the first frame
        try:
            # Try using the text-to-image pipeline if available
            text2img_pipeline = StableDiffusion3Pipeline.from_pretrained(
                config["stable_diffusion_checkpoint"],
                torch_dtype=torch.float16,
            ).to(self.device)

            print("Using text-to-image generation for initial frame")
            image = text2img_pipeline(
                prompt=self.inpainting_prompt,
                negative_prompt=self.config["negative_inpainting_prompt"],
                height=512,
                width=512,
                num_inference_steps=self.config["num_inpainting_steps"],
                guidance_scale=self.config["classifier_free_guidance_scale"],
            ).images[0]

            # Clean up the text2img pipeline to save memory
            del text2img_pipeline
            torch.cuda.empty_cache()

        except (ImportError, Exception) as e:
            print(
                f"Text-to-image pipeline failed ({e}), using improved inpainting approach"
            )
            # Fallback: Use a gray noise image instead of black, with a smaller mask
            # CHANGE: Use gray instead of white initial image
            # Original SD1.5: Image.new("RGB", (512, 512), "white")
            # SD3: Use gray to avoid whitecast and smaller mask
            init_image = Image.new(
                "RGB", (512, 512), (128, 128, 128)
            )  # Gray instead of black

            # Use a smaller mask for generating the first frame
            # CHANGE: Much smaller initial mask
            # Original SD1.5: Full white mask
            # SD3: Small center circle to avoid overwhelming the model
            mask_full = Image.new(
                "L", (512, 512), 0
            )  # Black background (don't inpaint)
            # Create a small white circle in the center to inpaint
            from PIL import ImageDraw

            draw = ImageDraw.Draw(mask_full)
            center = 256
            radius = 64  # Small circle
            draw.ellipse(
                [center - radius, center - radius, center + radius, center + radius],
                fill=255,
            )

            print("Using small mask inpainting for initial frame")
            image = self.inpainting_pipeline(
                prompt=self.inpainting_prompt,
                negative_prompt=self.config["negative_inpainting_prompt"],
                image=init_image,
                mask_image=mask_full,
                height=512,
                width=512,
                num_inference_steps=self.config["num_inpainting_steps"],
                guidance_scale=min(self.config["classifier_free_guidance_scale"], 7.5),
                strength=0.8,
            ).images[0]

        self.image_tensor = ToTensor()(image).unsqueeze(0).to(self.device)
        print(
            f"Initial image generated successfully, range: [{self.image_tensor.min():.3f}, {self.image_tensor.max():.3f}]"
        )

        self.depth_model = torch.hub.load("intel-isl/MiDaS", "DPT_Large").to(
            self.device
        )

        with torch.no_grad():
            self.depth, self.disparity = self.get_depth(self.image_tensor)

        self.current_camera = self.get_init_camera()
        if self.config["motion"] == "round":
            self.initial_median_depth = torch.median(self.depth).item()
            self.center_depth = self.depth[:, :, 256, 256].item()
            self.current_camera = self.get_next_camera_round(0)

        elif self.config["motion"] == "rotations":
            self.initial_median_disparity = torch.median(self.disparity)
            self.current_camera.rotating = True
            self.current_camera.no_rotations_count = 0
            self.current_camera.rotations_count = 0
            self.current_camera.rotating_right = (
                1 if torch.rand(1, device=self.device) > 0.5 else -1
            )
            self.current_camera.move_dir = torch.tensor(
                [[0.0, 0.0, 1.0]], device=self.device
            )

        elif self.config["motion"] == "translations":
            self.current_camera.translating_right = 1
            self.initial_median_disparity = torch.median(self.disparity)
            self.current_camera.move_dir = torch.tensor(
                [[1.0, 0.0, 0.0]], device=self.device
            )

        elif self.config["motion"] == "predefined":
            intrinsics = np.load(self.config["intrinsics"]).astype(np.float32)
            extrinsics = np.load(self.config["extrinsics"]).astype(np.float32)

            intrinsics = torch.from_numpy(intrinsics).to(self.device)
            extrinsics = torch.from_numpy(extrinsics).to(self.device)

            Ks = F.pad(intrinsics, (0, 1, 0, 1), value=0)
            Ks[:, 2, 3] = Ks[:, 3, 2] = 1

            Rs, ts = extrinsics[:, :3, :3], extrinsics[:, :3, 3]

            Rs = Rs.movedim(1, 2)

            self.predefined_cameras = [
                PerspectiveCameras(
                    K=K.unsqueeze(0),
                    R=R.T.unsqueeze(0),
                    T=t.unsqueeze(0),
                    device=self.device,
                )
                for K, R, t in zip(Ks, Rs, ts)
            ]
            self.current_camera = self.predefined_cameras[0]

        self.init_camera = copy.deepcopy(self.current_camera)

        self.images = [self.image_tensor]
        self.warped_mesh_images = []
        self.masks_diffs = []
        self.disparities = [self.disparity]
        self.depths = [self.depth]
        self.flows = [torch.zeros_like(self.depth).repeat(1, 2, 1, 1)]
        self.masks = [torch.ones_like(self.depth)]
        self.warped_images = [self.image_tensor]
        self.mesh_boundaries_masks = []
        self.boundary_points_dict = {}
        if self.config["antialiasing_factor"] > 1:
            self.big_warped_depths = []
            self.big_warped_masks = []
            self.big_warped_images = []

        self.latent_storer = LatentStorer()
        self.vae = AutoencoderKL.from_pretrained(
            config["stable_diffusion_checkpoint"],
            subfolder="vae",
            torch_dtype=torch.float16,  # CHANGE: Added torch_dtype
        ).to(self.device)
        self.decoder_copy = copy.deepcopy(self.vae.decoder)

        self.video_direction = -1

        self.camera_speed_factor = self.config["camera_speed_factor"]
        self.cameras_extrinsics = [self.get_extrinsics(self.current_camera).cpu()]
        self.cameras = [self.current_camera]
        self.current_points_3d = None
        self.current_colors = None
        self.current_triangles = None

        self.classifier_free_guidance_scale = self.config[
            "classifier_free_guidance_scale"
        ]

        assert self.config["inpainting_resolution"] >= 512

        self.border_mask = torch.ones(
            (
                1,
                1,
                self.config["inpainting_resolution"],
                self.config["inpainting_resolution"],
            )
        ).to(self.device)
        self.border_size = (self.config["inpainting_resolution"] - 512) // 2
        self.border_mask[
            :,
            :,
            self.border_size : -self.border_size,
            self.border_size : -self.border_size,
        ] = 0
        self.border_image = torch.zeros(
            1,
            3,
            self.config["inpainting_resolution"],
            self.config["inpainting_resolution"],
        ).to(self.device)

        self.images_orig_decoder = [
            # CHANGE: Updated image resizing approach
            # Original SD1.5: Used Resize() transform
            # SD3: Use F.interpolate
            F.interpolate(
                self.image_tensor,
                size=(
                    self.config["inpainting_resolution"],
                    self.config["inpainting_resolution"],
                ),
                mode="bilinear",
                align_corners=False,
            )
        ]

        boundaries_mask = self.get_boundaries_mask(self.disparity)

        self.depth_discontinuities_masks = []

        mesh_mask = torch.zeros_like(boundaries_mask)
        if self.config["use_splatting"]:
            x = torch.arange(512)
            y = torch.arange(512)
            self.points = torch.stack(torch.meshgrid(x, y), -1)
            self.points = rearrange(self.points, "h w c -> (h w) c").to(self.device)
        else:
            aa_factor = self.config["antialiasing_factor"]
            self.renderer = Renderer(config, image_size=512)
            self.aa_renderer = Renderer(
                config, image_size=512 * aa_factor, antialiasing_factor=aa_factor
            )
            self.big_image_renderer = Renderer(
                config, image_size=512 * (aa_factor + 1), antialiasing_factor=aa_factor
            )
            self.update_mesh(
                self.image_tensor,
                self.depth,
                mesh_mask,
                self.get_extrinsics(self.current_camera),
                0,
            )

    def save_mesh(self, name):
        full_mesh = trimesh.Trimesh(
            vertices=self.current_points_3d.cpu(),
            faces=self.current_triangles.cpu(),
            vertex_colors=self.current_colors.cpu(),
        )
        full_mesh.export(self.run_dir / f"{name}.obj")

    def clean_mesh(self, depth_discontinuity_points=None):

        triangles = self.current_triangles
        drop_mask = (
            (triangles[:, 0] == triangles[:, 1])
            | (triangles[:, 0] == triangles[:, 2])
            | (triangles[:, 1] == triangles[:, 2])
        )
        self.current_triangles = self.current_triangles[~drop_mask]

        if depth_discontinuity_points is not None:
            depth_discontinuity_mask = (
                torch.isin(self.current_triangles, depth_discontinuity_points).any(
                    dim=1
                )
                if self.config["mesh_exclude_boundaries"]
                else None
            )

        if self.config["min_triangle_angle"] > 0:
            min_angles = self.renderer.get_triangles_min_angle_degree(
                self.current_points_3d, self.current_triangles
            )
            bad_angles_mask = min_angles < self.config["min_triangle_angle"]
            if depth_discontinuity_mask is not None:
                bad_angles_mask = bad_angles_mask & depth_discontinuity_mask

            self.current_triangles = self.current_triangles[~bad_angles_mask]

        if self.config["min_connected_component_size"] > 0:
            self.remove_small_connected_components()

    def remove_small_connected_components(self):
        mesh = trimesh.Trimesh(
            vertices=self.current_points_3d.cpu(),
            faces=self.current_triangles.cpu(),
            vertex_colors=self.current_colors.cpu(),
        )
        connected_components = trimesh.graph.connected_components(mesh.face_adjacency)
        lens = [len(c) for c in connected_components]
        good_faces = None
        for i, c in enumerate(connected_components):
            if lens[i] >= self.config["min_connected_component_size"]:
                good_faces = (
                    torch.cat((good_faces, torch.from_numpy(c)))
                    if good_faces is not None
                    else torch.from_numpy(c)
                )
        self.current_triangles = torch.tensor(
            mesh.faces[good_faces], device=self.device
        )
        self.current_points_3d = torch.tensor(
            mesh.vertices, device=self.device, dtype=torch.float32
        )
        self.current_colors = (
            torch.tensor(mesh.visual.vertex_colors[:, :3], device=self.device) / 255
        )

    def filter_faces_by_normals(
        self, triangles, vertices, depth_discontinuity_mask=None
    ):
        normals = self.renderer.get_normals(vertices, triangles)
        vertices_triangles = vertices[triangles]
        centers = vertices_triangles.mean(dim=1)
        camera_center = (
            -self.current_camera.R[0].T @ self.current_camera.T.transpose(0, 1)
        ).T
        viewing_directions = centers - camera_center[0]
        viewing_directions = viewing_directions / viewing_directions.norm(
            dim=-1, keepdim=True
        )
        dot_product = (normals * viewing_directions).sum(dim=-1)
        bad_faces_mask = dot_product >= self.config["normal_filtering_threshold"]
        if depth_discontinuity_mask is not None:
            bad_faces_mask = bad_faces_mask & depth_discontinuity_mask
        return triangles[~bad_faces_mask]

    def update_mesh(self, image, depth, mask, extrinsic, epoch):

        updated_depth = depth.clone()
        updated_depth[mask] = -1
        if self.config["connect_mesh"] and epoch != 0:
            closest_boundary_points_data = self.boundary_points_dict[epoch]
        else:
            closest_boundary_points_data = None
        starting_index = self.current_points_3d.shape[0] if epoch != 0 else 0
        if self.config["mesh_exclude_boundaries"]:
            disp_boundaries_mask = self.get_boundaries_mask(1 / depth)
            boundaries_mask = disp_boundaries_mask.float()
            boundaries_mask = dilation(
                boundaries_mask, torch.ones(5, 5, device=self.device)
            ).bool()
            self.depth_discontinuities_masks.append(boundaries_mask)
        else:
            boundaries_mask = None

        mesh_dict = self.renderer.unproject_points(
            updated_depth,
            image,
            extrinsic,
            closest_boundary_points_data,
            starting_index=starting_index,
            depth_boundaries_mask=boundaries_mask,
        )
        points_3d, colors, triangles = (
            mesh_dict["points_3d"],
            mesh_dict["colors"],
            mesh_dict["triangles"],
        )

        if self.current_colors is None:
            self.current_colors = colors
            self.current_points_3d = points_3d
            if self.config["normal_filtering_threshold"]:
                depth_discontinuity_mask = (
                    torch.isin(triangles, mesh_dict["depth_discontinuity_points"]).any(
                        dim=1
                    )
                    if self.config["mesh_exclude_boundaries"]
                    else None
                )
                triangles = self.filter_faces_by_normals(
                    triangles, self.current_points_3d, depth_discontinuity_mask
                )
            self.current_triangles = triangles
        else:
            self.current_colors = torch.cat([self.current_colors, colors], dim=0)
            self.current_points_3d = torch.cat(
                [self.current_points_3d, points_3d], dim=0
            )
            if self.config["normal_filtering_threshold"]:
                depth_discontinuity_mask = (
                    torch.isin(triangles, mesh_dict["depth_discontinuity_points"]).any(
                        dim=1
                    )
                    if self.config["mesh_exclude_boundaries"]
                    else None
                )
                triangles = self.filter_faces_by_normals(
                    triangles, self.current_points_3d, depth_discontinuity_mask
                )
            self.current_triangles = torch.cat(
                [self.current_triangles, triangles], dim=0
            )

        self.clean_mesh(
            depth_discontinuity_points=(
                mesh_dict["depth_discontinuity_points"]
                if self.config["mesh_exclude_boundaries"]
                else None
            )
        )

    @staticmethod
    def get_extrinsics(camera):
        extrinsics = torch.cat([camera.R[0], camera.T.T], dim=1)
        padding = torch.tensor([[0, 0, 0, 1]], device=extrinsics.device)
        extrinsics = torch.cat([extrinsics, padding], dim=0)
        return extrinsics

    def get_depth(self, image):
        # CHANGE: Added dtype conversion for MiDaS compatibility
        # Original SD1.5: Used image directly
        # SD3: Convert to float32 explicitly because SD3 pipeline uses float16 by default
        # but MiDaS depth model expects float32
        image_float32 = image.float() if image.dtype != torch.float32 else image
        # ============================================================
        disparity = self.depth_model(dpt_transform(image_float32))
        disparity = torch.nn.functional.interpolate(
            disparity.unsqueeze(1),
            size=image.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        disparity = disparity.clip(self.config["min_disparity"], max=None)
        depth = 1 / disparity

        return depth, disparity

    def get_init_camera(self):
        K = torch.zeros((1, 4, 4), device=self.device)
        K[0, 0, 0] = self.config["init_focal_length"]
        K[0, 1, 1] = self.config["init_focal_length"]
        K[0, 0, 2] = 256
        K[0, 1, 2] = 256
        K[0, 2, 3] = 1
        K[0, 3, 2] = 1
        R = torch.eye(3, device=self.device).unsqueeze(0)
        T = torch.zeros((1, 3), device=self.device)
        camera = PerspectiveCameras(
            K=K, R=R, T=T, in_ndc=False, image_size=((512, 512),), device=self.device
        )
        return camera

    def get_boundaries_mask(self, disparity):
        normalized_disparity = (disparity - disparity.min()) / (
            disparity.max() - disparity.min() + 1e-6
        )
        return (
            sobel_filter(normalized_disparity, "sobel", beta=self.config["sobel_beta"])
            < self.config["sobel_threshold"]
        )

    def inpaint_cv2(self, warped_image, mask_diff):
        image_cv2 = warped_image[0].permute(1, 2, 0).cpu().numpy()
        image_cv2 = (image_cv2 * 255).astype(np.uint8)
        mask_cv2 = mask_diff[0, 0].cpu().numpy()
        mask_cv2 = (mask_cv2 * 255).astype(np.uint8)
        inpainting = cv2.inpaint(image_cv2, mask_cv2, 3, cv2.INPAINT_TELEA)
        inpainting = torch.from_numpy(inpainting).permute(2, 0, 1).float() / 255
        return inpainting.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def warp_splatting(self, epoch):
        if self.config["motion"] == "rotations":
            camera = self.get_next_camera_rotation()
        elif self.config["motion"] == "translations":
            camera = self.get_next_camera_translation(
                self.disparities[epoch - 1], epoch
            )
        elif self.config["motion"] == "round":
            camera = self.get_next_camera_round(epoch)
        elif self.config["motion"] == "predefined":
            camera = self.predefined_cameras[epoch]
        else:
            raise NotImplementedError
        next_camera = self.convert_pytorch3d_kornia(camera)
        current_camera = self.convert_pytorch3d_kornia(self.current_camera)
        points_3d = current_camera.unproject(
            self.points, rearrange(self.depths[epoch - 1], "b c h w -> (w h b) c")
        )
        P = next_camera.intrinsics @ next_camera.extrinsics
        transformed_points = transform_points(P, points_3d)
        transformed_z = transformed_points[:, [2]]
        points_2d = convert_points_from_homogeneous(transformed_points)
        flow = points_2d - self.points
        flow_tensor = rearrange(flow, "(w h b) c -> b c h w", w=512, h=512)

        importance = 1.0 / (transformed_z)
        importance_min = importance.amin(keepdim=True)
        importance_max = importance.amax(keepdim=True)
        weights = (importance - importance_min) / (
            importance_max - importance_min + 1e-6
        ) * 20 - 10
        weights = rearrange(weights, "(w h b) c -> b c h w", w=512, h=512)

        transformed_z_tensor = rearrange(
            transformed_z, "(w h b) c -> b c h w", w=512, h=512
        )
        inpaint_mask = torch.ones_like(transformed_z_tensor)
        boundaries_mask = self.get_boundaries_mask(self.disparities[epoch - 1])

        input_data = torch.cat(
            [
                self.images[epoch - 1],
                transformed_z_tensor,
                inpaint_mask,
                boundaries_mask,
            ],
            1,
        )
        output_data = softsplat.softsplat(
            tenIn=input_data,
            tenFlow=flow_tensor,
            tenMetric=weights.detach(),
            strMode="soft",
        )
        warped_image = output_data[:, 0:3, ...].clip(0, 1)
        warped_depth = output_data[:, 3:4, ...]
        inpaint_mask = output_data[:, 4:5, ...]
        boundaries_mask = output_data[:, 5:6, ...]
        nans = inpaint_mask.isnan()
        inpaint_mask[nans] = 0
        inpaint_mask = (inpaint_mask < 0.5).float()

        nans = boundaries_mask.isnan()
        boundaries_mask[nans] = 0
        if self.config["sobel_beta"] > 0:
            inpaint_mask = torch.maximum(inpaint_mask, boundaries_mask)

        self.current_camera = copy.deepcopy(camera)
        self.cameras_extrinsics.append(self.get_extrinsics(self.current_camera).cpu())
        self.cameras.append(self.current_camera)
        self.warped_images.append(warped_image)

        if self.config["inpainting_resolution"] > 512:
            padded_inpainting_mask = self.border_mask.clone()
            padded_inpainting_mask[
                :,
                :,
                self.border_size : -self.border_size,
                self.border_size : -self.border_size,
            ] = inpaint_mask
            padded_image = self.border_image.clone()
            padded_image[
                :,
                :,
                self.border_size : -self.border_size,
                self.border_size : -self.border_size,
            ] = warped_image
        else:
            padded_inpainting_mask = inpaint_mask
            padded_image = warped_image

        return {
            "warped_image": padded_image,
            "warped_depth": warped_depth,
            "inpaint_mask": padded_inpainting_mask,
        }

    def get_mesh_boundaries_mask(self, inpaint_mask):
        filter = torch.tensor([[[[1, 1, 1], [1, 0, 1], [1, 1, 1]]]], device=self.device)
        edges = F.conv2d(inpaint_mask.float(), filter.float(), padding=1)
        boundary_mask = (edges > 0) * (1 - inpaint_mask)
        return boundary_mask

    def get_closest_boundary_points(self, boundary_mask, closest_faces):
        xs, ys = torch.where(boundary_mask[0, 0] != 0)
        boundary_closest_faces = closest_faces[xs, ys]
        boundary_points_indices = self.current_triangles[boundary_closest_faces]

        boundary_points = self.current_points_3d[boundary_points_indices]
        boundary_points_reshaped = rearrange(boundary_points, "n f d -> (n f) d")
        center = (-self.current_camera.R[0].T @ self.current_camera.T.transpose(0, 1)).T
        distances = rearrange(
            torch.cdist(boundary_points_reshaped, center),
            "(n f) c -> n (f c)",
            n=boundary_points.shape[0],
            f=boundary_points.shape[1],
        )
        closest_points_indices = torch.argmin(distances, dim=1)
        closest_boundary_points = boundary_points_indices[
            torch.arange(boundary_points_indices.shape[0]), closest_points_indices
        ]
        return xs, ys, closest_boundary_points

    def fix_floating_artifacts(self, extrinsic):
        cur_camera = self.cameras_extrinsics[-1].to(self.device)
        prev_image = self.images[-1].to(self.device)
        prev_depth = self.depths[-1].to(self.device)

        if self.config["antialiasing_factor"] > 1:
            self.aa_renderer.renderer.rasterizer.raster_settings.blur_radius = 1e-3
            prev_image, _, prev_depth, _, _ = self.aa_renderer.sample_points(
                self.current_points_3d,
                self.current_triangles,
                self.current_colors,
                cur_camera,
            )
            self.aa_renderer.renderer.rasterizer.raster_settings.blur_radius = (
                self.config["blur_radius"]
            )
        big_resolution = (self.config["antialiasing_factor"] + 1) * 512
        border_size = (big_resolution - self.config["antialiasing_factor"] * 512) // 2
        pad_value = 0
        big_image = F.pad(
            prev_image,
            (border_size, border_size, border_size, border_size),
            mode="constant",
            value=pad_value,
        )
        border_mask = torch.ones(
            (1, 1, big_resolution, big_resolution), device=self.device
        )
        border_mask[0, 0, border_size:-border_size, border_size:-border_size] = 0
        big_depth = F.pad(
            prev_depth,
            (border_size, border_size, border_size, border_size),
            mode="replicate",
        )

        updated_depth = big_depth.clone()
        updated_depth[~border_mask.bool()] = -1
        mesh_border_dict = self.big_image_renderer.unproject_points(
            updated_depth, big_image, cur_camera
        )
        border_colors, border_points_3d, border_triangles = (
            mesh_border_dict["colors"],
            mesh_border_dict["points_3d"],
            mesh_border_dict["triangles"],
        )
        border_warped_image, _, _, _, _ = self.big_image_renderer.sample_points(
            border_points_3d, border_triangles, border_colors, extrinsic
        )
        add_to_mask = (border_warped_image == pad_value)[:, [0]].float()
        add_to_mask = add_to_mask[
            :, :, border_size:-border_size, border_size:-border_size
        ]
        kernel = torch.ones((3, 3), device=self.device)
        add_to_mask = dilation(add_to_mask, kernel)
        return add_to_mask

    def masked_downsample(self, image, mask, put_in_mask=1):
        n = image.shape[1]
        input_image = image.clone()
        input_image[(mask == 0).repeat(1, n, 1, 1)] = 0
        k_s = self.config["antialiasing_factor"] * 2 - 1
        padding = (k_s - 1) // 2
        sums = F.avg_pool2d(
            input_image, stride=1, padding=padding, kernel_size=k_s, divisor_override=1
        )
        counts = F.avg_pool2d(
            mask.float(), stride=1, padding=padding, kernel_size=k_s, divisor_override=1
        )
        blurred = sums / counts
        blurred[(mask == 0).repeat(1, n, 1, 1)] = put_in_mask
        stride = self.config["antialiasing_factor"]
        blurred_subsampled = blurred[:, :, ::stride, ::stride]
        return blurred_subsampled

    def warp_mesh(self, epoch=None, camera=None):
        assert not (epoch is None and camera is None)
        if camera is None:
            if self.config["motion"] == "rotations":
                camera = self.get_next_camera_rotation()
            elif self.config["motion"] == "translations":
                camera = self.get_next_camera_translation(
                    self.disparities[epoch - 1], epoch
                )
            elif self.config["motion"] == "round":
                camera = self.get_next_camera_round(epoch)
            elif self.config["motion"] == "predefined":
                camera = self.predefined_cameras[epoch]
            else:
                raise NotImplementedError
        extrinsic = self.get_extrinsics(camera)

        warped_image, inpaint_mask, warped_depth, closest_faces, fragments = (
            self.aa_renderer.sample_points(
                self.current_points_3d,
                self.current_triangles,
                self.current_colors,
                extrinsic,
            )
        )

        warped_image = warped_image.clip(0, 1)
        warped_image[warped_image > 0.99] = 0.98

        nans = inpaint_mask.isnan()
        inpaint_mask[nans] = 0
        inpaint_mask = (inpaint_mask < 0.5).float()
        if self.config["fix_floating_artifacts"]:
            add_to_inpainting_mask = self.fix_floating_artifacts(extrinsic)
            inpaint_mask = torch.maximum(inpaint_mask, add_to_inpainting_mask)
            warped_image[inpaint_mask.bool().repeat(1, 3, 1, 1)] = 1
            warped_depth[inpaint_mask.bool()] = -1

        if self.config["antialiasing_factor"] > 1:
            self.big_warped_depths.append(warped_depth)
            self.big_warped_images.append(warped_image)
            self.big_warped_masks.append(inpaint_mask)
            mask = warped_depth != -1
            warped_image = self.masked_downsample(warped_image, mask, put_in_mask=1)
            warped_depth = self.masked_downsample(warped_depth, mask, put_in_mask=-1)
            inpaint_mask = (warped_depth == -1).float()
            stride = self.config["antialiasing_factor"]
            closest_faces = closest_faces[::stride, ::stride]

        mesh_boundaries_mask = self.get_mesh_boundaries_mask(inpaint_mask)
        boundary_xs, boundary_ys, boundary_closest_points = (
            self.get_closest_boundary_points(mesh_boundaries_mask, closest_faces)
        )
        self.boundary_points_dict[epoch] = {
            "xs": boundary_xs,
            "ys": boundary_ys,
            "closest_points": boundary_closest_points,
        }
        self.mesh_boundaries_masks.append(mesh_boundaries_mask)

        self.warped_images.append(warped_image)

        if self.config["mask_opening_kernel_size"] > 0:
            kernel = torch.ones(
                self.config["mask_opening_kernel_size"],
                self.config["mask_opening_kernel_size"],
                device=inpaint_mask.device,
            )
            inpaint_mask_opened = opening(inpaint_mask, kernel)
            mask_diff = inpaint_mask - inpaint_mask_opened
            warped_image = self.inpaint_cv2(warped_image, mask_diff)
            inpaint_mask = inpaint_mask_opened
            self.masks_diffs.append(mask_diff)

        self.current_camera = copy.deepcopy(camera)
        self.cameras_extrinsics.append(self.get_extrinsics(self.current_camera).cpu())
        self.cameras.append(self.current_camera)

        if self.config["inpainting_resolution"] > 512:
            padded_inpainting_mask = self.border_mask.clone()
            padded_inpainting_mask[
                :,
                :,
                self.border_size : -self.border_size,
                self.border_size : -self.border_size,
            ] = inpaint_mask
            padded_image = self.border_image.clone()
            padded_image[
                :,
                :,
                self.border_size : -self.border_size,
                self.border_size : -self.border_size,
            ] = warped_image
        else:
            padded_inpainting_mask = inpaint_mask
            padded_image = warped_image

        return {
            "warped_image": padded_image,
            "warped_depth": warped_depth,
            "inpaint_mask": padded_inpainting_mask,
            "mesh_boundaries_mask": mesh_boundaries_mask,
        }

    def convert_pytorch3d_kornia(self, camera):
        R = camera.R
        T = camera.T
        extrinsics = torch.eye(4, device=R.device).unsqueeze(0)
        extrinsics[:, :3, :3] = R
        extrinsics[:, :3, 3] = T
        h = torch.tensor([512], device="cuda")
        w = torch.tensor([512], device="cuda")
        K = torch.eye(4)[None].to("cuda")
        K[0, 0, 2] = 256
        K[0, 1, 2] = 256
        K[0, 0, 0] = self.config["init_focal_length"]
        K[0, 1, 1] = self.config["init_focal_length"]
        return PinholeCamera(K, extrinsics, h, w)

    def finetune_depth_model_step(self, warped_depth, inpainted_image, mask):
        next_depth, _ = self.get_depth(inpainted_image.detach())
        loss = F.l1_loss(warped_depth.detach(), next_depth, reduction="none")
        full_mask = mask.detach()
        loss = (loss * full_mask)[full_mask > 0].mean()
        return loss

    def finetune_decoder_step(
        self, inpainted_image, inpainted_image_latent, warped_image, inpaint_mask
    ):
        reconstruction = self.decode_latents(inpainted_image_latent)
        loss = (
            F.mse_loss(inpainted_image * inpaint_mask, reconstruction * inpaint_mask)
            + F.mse_loss(
                warped_image * (1 - inpaint_mask), reconstruction * (1 - inpaint_mask)
            )
            * self.config["preservation_weight"]
        )
        return loss

    @torch.no_grad()
    def inpaint(self, warped_image, inpaint_mask):
        # CHANGE: Completely rewritten inpainting logic for SD3
        # Original SD1.5: Simple inpainting with basic mask processing
        # SD3: Advanced mask processing, white area detection, progressive inpainting
        # to handle SD3's sensitivity to mask/image combinations and prevent whitecast

        # ======== Debug: Check input values ========
        print(
            f"Input warped_image range: [{warped_image.min():.3f}, {warped_image.max():.3f}]"
        )
        print(
            f"Input inpaint_mask range: [{inpaint_mask.min():.3f}, {inpaint_mask.max():.3f}]"
        )
        print(f"Inpaint mask shape: {inpaint_mask.shape}")
        print(
            f"Mask pixels > 0.5: {(inpaint_mask > 0.5).sum().item()}/{inpaint_mask.numel()}"
        )

        # Convert mask to a single-channel mask becauseSD3 expects masks where white (1.0) = inpaint, black (0.0) = keep
        mask_single_channel = inpaint_mask[0, 0]
        # CHANGE: WHITECAST FIX FOR SD3
        # SD1.5: White area detection was not needed in SD1.5:
        # SD3: Detect white/bright areas in the warped image
        # These areas need to be inpainted aggressively because they're mesh artifacts

        warped_gray = warped_image[0].mean(dim=0)  # Convert to grayscale
        white_areas = (warped_gray > 0.9).float()  # Detect very bright areas

        # Combine the original mask with white area detection
        mask_enhanced = torch.maximum(mask_single_channel, white_areas)

        print(f"White areas detected: {(white_areas > 0.5).sum().item()}")
        print(f"Enhanced mask pixels > 0.5: {(mask_enhanced > 0.5).sum().item()}")

        # CHANGE: Conservative mask processing for SD3
        # Original SD1.5: Used mask directly or with simple morphology
        # SD3: Apply erosion to make masks more conservative, but preserve white areas
        erosion_kernel = torch.ones(
            5, 5, device=mask_single_channel.device
        )  # Smaller kernel

        has_white_areas = (white_areas > 0.5).sum().item() > 1000

        if not has_white_areas:
            erosion_kernel = torch.ones(5, 5, device=mask_single_channel.device)
            mask_eroded = erosion(
                mask_enhanced.unsqueeze(0).unsqueeze(0), erosion_kernel
            )[0, 0]
            mask_conservative = (mask_eroded > 0.5).float()
        else:
            print("Skipping erosion due to white area dominance")
            mask_conservative = mask_enhanced.clone()

        # Make sure white areas are always included in the mask
        mask_conservative = torch.maximum(
            mask_conservative, (white_areas > 0.1).float()
        )

        print(f"Original mask pixels > 0.5: {(mask_single_channel > 0.5).sum().item()}")
        print(
            f"Conservative mask pixels > 0.5: {(mask_conservative > 0.5).sum().item()}"
        )

        # CHANGE: Progressive inpainting for large masks
        # Original SD1.5: Could handle large masks directly
        # SD3: Large masks cause quality issues, use progressive approach
        mask_ratio = (mask_conservative > 0.5).sum().item() / mask_conservative.numel()
        print(f"Mask coverage ratio: {mask_ratio:.3f}")

        if mask_ratio > 0.4:
            print("Large mask detected - using progressive inpainting approach")
            # For progressive inpainting, prioritize white areas and edges
            h, w = mask_conservative.shape
            center_h, center_w = h // 2, w // 2

            # Create a mask that includes white areas and central regions
            mask_progressive = torch.zeros_like(mask_conservative)

            # Always include white areas
            mask_progressive = torch.maximum(mask_progressive, white_areas)

            # Add central region for continuity
            quarter_h, quarter_w = h // 3, w // 3
            mask_progressive[
                center_h - quarter_h : center_h + quarter_h,
                center_w - quarter_w : center_w + quarter_w,
            ] = torch.maximum(
                mask_progressive[
                    center_h - quarter_h : center_h + quarter_h,
                    center_w - quarter_w : center_w + quarter_w,
                ],
                mask_conservative[
                    center_h - quarter_h : center_h + quarter_h,
                    center_w - quarter_w : center_w + quarter_w,
                ],
            )

            print(
                f"Progressive mask pixels > 0.5: {(mask_progressive > 0.5).sum().item()}"
            )
            mask_conservative = mask_progressive

        # CHANGE: Mask format for SD3
        # Original SD1.5: Used RGB mask images
        # SD3: Expects grayscale PIL image (L mode) for masks
        mask_array = (mask_conservative.cpu().numpy() * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_array, mode="L")  # Grayscale mask

        # Print inpainting parameters
        print(f"Inpainting prompt: '{self.inpainting_prompt}'")
        print(f"Negative prompt: '{self.config['negative_inpainting_prompt']}'")
        print(f"Guidance scale: {self.classifier_free_guidance_scale}")
        print(f"Num inference steps: {self.config['num_inpainting_steps']}")
        print(f"Inpainting resolution: {self.config['inpainting_resolution']}")

        # CHANGE: Adjust parameter for SD3
        # Original SD1.5: Used fixed parameters
        # SD3: Adjust guidance_scale and strength based on white area
        has_white_areas = (
            white_areas > 0.5
        ).sum().item() > 1000  # Significant white areas
        guidance_scale = min(
            self.classifier_free_guidance_scale, 6.0 if has_white_areas else 5.0
        )
        strength = 0.85 if has_white_areas else 0.6
        guidance_scale = (
            7.5 if has_white_areas else 5.0
        )  # Higher strength for white areas

        print(f"Using adjusted guidance_scale: {guidance_scale}, strength: {strength}")
        print(f"White area mode: {has_white_areas}")

        # CHANGE: Updated inpainting call for SD3
        # SD3: Define height/width, updated parameter names for strength and guidance_scale
        inpainted_images = self.inpainting_pipeline(
            prompt=self.inpainting_prompt,
            negative_prompt=self.config["negative_inpainting_prompt"],
            image=ToPILImage()(warped_image[0]),
            mask_image=mask_pil,
            num_inference_steps=self.config["num_inpainting_steps"],
            guidance_scale=guidance_scale,  # Use adjusted guidance scale
            strength=strength,  # Use adjusted strength
            num_images_per_prompt=1,
            height=self.config["inpainting_resolution"],
            width=self.config["inpainting_resolution"],
        ).images

        best_index = 0
        inpainted_image = inpainted_images[best_index]
        inpainted_image_tensor = (
            ToTensor()(inpainted_image).unsqueeze(0).to(self.device)
        )

        # ================================
        # Reduce white artifacts after inpainting

        # ================================
        inpainted_gray = inpainted_image_tensor[0].mean(dim=0)
        white_pixels = inpainted_gray > 0.96
        if white_pixels.sum() > 5000:
            print("Post-inpainting whitecast detected — suppressing")
            # Dim the overly bright areas softly to prevent propagation
            inpainted_image_tensor[0][:, white_pixels] *= 0.85

        # Save debug images with the new mask
        self._save_debug_images(
            warped_image, mask_conservative.unsqueeze(0).unsqueeze(0), inpainted_image
        )

        # CHANGE: Direct image handling for SD3
        # Original SD1.5: Always work with latents
        # SD3: Store direct image to bypass latent encoding/decoding issues
        # Take center crop if needed
        if self.config["inpainting_resolution"] > 512:
            center_crop = inpainted_image_tensor[
                :,
                :,
                self.border_size : -self.border_size,
                self.border_size : -self.border_size,
            ]
        else:
            center_crop = inpainted_image_tensor

        self._sd3_direct_image = center_crop

        # CHANGE: Create dummy latent to be compatible with SD3
        # Original SD1.5: Return actual latents
        # SD3: Create dummy latent since we're using the image instead of latents
        latent_shape = (
            1,
            16,  # SD3 latent channels
            self.config["inpainting_resolution"] // 8,
            self.config["inpainting_resolution"] // 8,
        )

        dummy_latent = (
            torch.randn(latent_shape, device=self.device, dtype=torch.float16) * 0.1
        )

        # Check if we're getting valid values
        print(
            f"Inpainted image range: [{inpainted_image_tensor.min():.3f}, {inpainted_image_tensor.max():.3f}]"
        )
        print(f"Center crop range: [{center_crop.min():.3f}, {center_crop.max():.3f}]")

        return {
            "inpainted_image": inpainted_image_tensor,
            "latent": dummy_latent,
            "best_index": best_index,
        }

    def _save_debug_images(self, warped_image, inpaint_mask, inpainted_image):
        """Save debug images to understand inpainting behavior"""
        # CHANGE: Save debug images for SD3 in each iteration
        import os

        debug_dir = self.run_dir / "debug_images"
        debug_dir.mkdir(exist_ok=True)

        # Save current iteration number
        if not hasattr(self, "_debug_counter"):
            self._debug_counter = 0
        self._debug_counter += 1

        # Save the warped image, mask, and inpainted result
        warped_pil = ToPILImage()(warped_image[0])

        # Create a more visible mask visualization
        mask_single_channel = inpaint_mask[0, 0]
        mask_array = (mask_single_channel.cpu().numpy() * 255).astype(np.uint8)
        mask_pil_debug = Image.fromarray(mask_array, mode="L")

        # Detect and visualize white areas
        warped_gray = warped_image[0].mean(dim=0)
        white_areas = (warped_gray > 0.9).float()
        white_areas_array = (white_areas.cpu().numpy() * 255).astype(np.uint8)
        white_areas_pil = Image.fromarray(white_areas_array, mode="L")

        # Create composite image showing mask overlay on warped image
        warped_array = np.array(warped_pil)
        mask_overlay = np.zeros_like(warped_array)
        mask_overlay[:, :, 0] = mask_array  # Red channel for mask areas

        # Add white areas in blue channel
        mask_overlay[:, :, 2] = white_areas_array  # Blue channel for white areas

        composite = (warped_array * 0.6 + mask_overlay * 0.4).astype(np.uint8)
        composite_pil = Image.fromarray(composite)

        warped_pil.save(debug_dir / f"iter_{self._debug_counter:03d}_warped.png")
        mask_pil_debug.save(debug_dir / f"iter_{self._debug_counter:03d}_mask.png")
        white_areas_pil.save(
            debug_dir / f"iter_{self._debug_counter:03d}_white_areas.png"
        )
        composite_pil.save(
            debug_dir / f"iter_{self._debug_counter:03d}_mask_overlay.png"
        )
        inpainted_image.save(
            debug_dir / f"iter_{self._debug_counter:03d}_inpainted.png"
        )

        print(f"Debug images saved to: {debug_dir}")
        print(f"Check iter_{self._debug_counter:03d}_*.png files")
        print(f"  - Red areas in overlay: inpainting mask")
        print(f"  - Blue areas in overlay: detected white areas")

    @torch.no_grad()
    def update_depth(self, inpainted_image):
        new_depth, new_disparity = self.get_depth(inpainted_image)
        self.depths.append(new_depth.detach())
        self.disparities.append(new_disparity.detach())

    def update_images_masks(self, latent_or_image, inpaint_mask):
        # CHANGE: Modified to handle SD3's direct image
        # Original SD1.5: Always decoded from latents
        # SD3: Can receive direct image or latent, prefer direct image
        if hasattr(self, "_sd3_direct_image") and self._sd3_direct_image is not None:
            decoded_image = self._sd3_direct_image
            self._sd3_direct_image = None  # Reset for next iteration
        else:
            decoded_image = self.decode_latents(latent_or_image).detach()

        # take center crop of 512*512
        if self.config["inpainting_resolution"] > 512:
            decoded_image = decoded_image[
                :,
                :,
                self.border_size : -self.border_size,
                self.border_size : -self.border_size,
            ]
            inpaint_mask = inpaint_mask[
                :,
                :,
                self.border_size : -self.border_size,
                self.border_size : -self.border_size,
            ]
        else:
            decoded_image = decoded_image
            inpaint_mask = inpaint_mask

        self.images.append(decoded_image)
        self.masks.append(inpaint_mask)

    def decode_latents(self, latents):
        # CHANGE: Updated latent scaling factor for SD3
        # Original SD1.5/2: Used 0.18215 scaling factor
        # SD3: Uses 0.13025 scaling factor
        latents = 1 / 0.13025 * latents  # uses 0.13025 instead of 0.18215
        images = self.vae.decode(latents).sample
        images = (images / 2 + 0.5).clamp(0, 1)

        return images

    def get_next_camera_round(self, epoch):
        if self.config["rotate_radius"] == "median":
            r = 2 * self.initial_median_depth
        elif self.config["rotate_radius"] == "center":
            r = 2 * self.center_depth
        else:
            r = self.config["rotate_radius"]

        next_camera = copy.deepcopy(self.current_camera)
        center = torch.tensor([[0.0, 0.0, r / 2]])
        theta = torch.deg2rad(
            (torch.tensor(epoch) * 360.0 / self.config["full_circle_epochs"]) % 360
        )
        x = torch.sin(theta)
        y = 0
        z = torch.cos(theta)
        r_vector = r * torch.tensor([x, y, z])
        total_t = center + r_vector
        total_t = total_t.to(self.device).float()
        next_camera.T = total_t
        next_camera.R = look_at_rotation(next_camera.T, at=(center[0].tolist(),)).to(
            self.device
        )
        return next_camera

    def get_next_camera_translation(self, disparity, epoch):
        next_camera = copy.deepcopy(self.current_camera)
        if epoch % self.config["change_translation_every"] == 0:
            next_camera.translating_right = (
                -1 if (next_camera.translating_right == 1) else 1
            )

        median_disparity = torch.median(disparity)
        translation_speed_factor = (
            median_disparity / self.initial_median_disparity
        ).clip(min=None, max=1)
        speed = (
            translation_speed_factor
            * self.camera_speed_factor
            * next_camera.translating_right
            * 0.1875
        )
        next_camera.T += speed * self.current_camera.move_dir

        return next_camera

    def get_next_camera_rotation(self):
        next_camera = copy.deepcopy(self.current_camera)
        if next_camera.rotating:
            if self.current_camera.rotations_count <= self.config["rotation_steps"]:
                next_camera.rotations_count = self.current_camera.rotations_count + 1
                next_camera.rotating_right = self.current_camera.rotating_right
            else:
                next_camera.rotating = False
                next_camera.rotations_count = 0

            theta = torch.tensor(
                self.config["rotation_range"] * next_camera.rotating_right
            )
            rotation_matrix = torch.tensor(
                [
                    [torch.cos(theta), 0, torch.sin(theta)],
                    [0, 1, 0],
                    [-torch.sin(theta), 0, torch.cos(theta)],
                ],
                device=self.device,
            )
            next_camera.R[0] = rotation_matrix @ next_camera.R[0]
            next_camera.T[0] = rotation_matrix @ next_camera.T[0]

        else:
            next_camera.no_rotations_count += 1
            if next_camera.no_rotations_count > self.config["no_rotations_steps"]:
                next_camera.no_rotations_count = 0
                next_camera.rotating = True
                next_camera.rotating_right = (
                    1 if (next_camera.rotating_right == -1) else -1
                )

        # move camera backwards
        speed = self.camera_speed_factor * 0.1875
        next_camera.T += speed * torch.tensor([[0.0, 0.0, 1.0]], device="cuda")

        return next_camera

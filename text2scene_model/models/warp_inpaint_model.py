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
    StableDiffusion3ControlNetInpaintingPipeline,
    StableDiffusion3Pipeline,
    DPMSolverMultistepScheduler,
    AutoencoderKL,
    SD3ControlNetModel,
)
from einops import rearrange
from kornia.geometry import (
    PinholeCamera,
    transform_points,
    convert_points_from_homogeneous,
)
from kornia.morphology import dilation, opening
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
        self.size = config["inpainting_resolution"]
        super().__init__()
        if config["use_splatting"]:
            sys.path.append("util/softmax-splatting")
            import softsplat

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

        # ====================== Pipeline Loading ======================

        controlnet = SD3ControlNetModel.from_pretrained(
            "../sd3con",
            torch_dtype=torch.float16,
        )

        self.inpainting_pipeline = (
            StableDiffusion3ControlNetInpaintingPipeline.from_pretrained(
                self.config["stable_diffusion_checkpoint"],
                torch_dtype=torch.float16,
                controlnet=controlnet,
            )
        )

        self.inpainting_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            self.inpainting_pipeline.scheduler.config
        )
        self.inpainting_pipeline = self.inpainting_pipeline.to(self.device)

        if self.config["use_xformers"]:
            self.inpainting_pipeline.set_use_memory_efficient_attention_xformers(True)

        # ====================== Initial Image Generation ======================
        # Use the DreamBooth SD3 pipeline for the first image generation (text-to-image, no mask)
        first_txt2img_pipeline = StableDiffusion3Pipeline.from_pretrained(
            self.config["stable_diffusion_checkpoint"], torch_dtype=torch.float16
        ).to(self.device)

        generator = torch.Generator(device=self.device).manual_seed(
            self.config.get("seed", 42)
        )

        image = first_txt2img_pipeline(
            prompt=self.config["inpainting_prompt"],
            negative_prompt=self.config.get("negative_inpainting_prompt", None),
            num_inference_steps=self.config["num_inpainting_steps"],
            guidance_scale=self.config["classifier_free_guidance_scale"],
            generator=generator,
            height=self.size,
            width=self.size,
        ).images[0]
        self.image_tensor = ToTensor()(image).unsqueeze(0).to(self.device)

        # ====================== Depth Estimation ======================
        # Use MiDaS for monocular depth estimation
        self.depth_model = torch.hub.load("intel-isl/MiDaS", "DPT_Large").to(
            self.device
        )

        with torch.no_grad():
            self.depth, self.disparity = self.get_depth(self.image_tensor)

        # ====================== Camera Initialization ======================
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

            # Extend intrinsics to 4x4 with zeros and assign 1 to the last row and column as required by the camera class
            Ks = F.pad(intrinsics, (0, 1, 0, 1), value=0)
            Ks[:, 2, 3] = Ks[:, 3, 2] = 1

            Rs, ts = extrinsics[:, :3, :3], extrinsics[:, :3, 3]

            # PerspectiveCameras operate on row-vector matrices while the loaded extrinsics are column-vector matrices
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
            config["stable_diffusion_checkpoint"], subfolder="vae"
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
        self.images_orig_decoder = [Resize((self.size, self.size))(self.image_tensor)]

        boundaries_mask = self.get_boundaries_mask(self.disparity)

        self.depth_discontinuities_masks = []

        mesh_mask = torch.zeros_like(boundaries_mask)
        if self.config["use_splatting"]:
            x = torch.arange(self.size)
            y = torch.arange(self.size)
            self.points = torch.stack(torch.meshgrid(x, y), -1)
            self.points = rearrange(self.points, "h w c -> (h w) c").to(self.device)
        else:
            aa_factor = self.config["antialiasing_factor"]

            self.renderer = Renderer(config, image_size=self.size)
            self.aa_renderer = Renderer(
                config, image_size=self.size * aa_factor, antialiasing_factor=aa_factor
            )
            self.big_image_renderer = Renderer(
                config,
                image_size=self.size * (aa_factor + 1),
                antialiasing_factor=aa_factor,
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
        # remove triangles with duplicate vertices
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
        # remove stretched triangles
        if self.config["min_triangle_angle"] > 0:
            min_angles = self.renderer.get_triangles_min_angle_degree(
                self.current_points_3d, self.current_triangles
            )
            bad_angles_mask = min_angles < self.config["min_triangle_angle"]
            if depth_discontinuity_mask is not None:
                bad_angles_mask = bad_angles_mask & depth_discontinuity_mask

            self.current_triangles = self.current_triangles[~bad_angles_mask]

        # remove small connected components
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
        disparity = self.depth_model(dpt_transform(image))
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
        K[0, 0, 2] = self.size // 2
        K[0, 1, 2] = self.size // 2
        K[0, 2, 3] = 1
        K[0, 3, 2] = 1
        R = torch.eye(3, device=self.device).unsqueeze(0)
        T = torch.zeros((1, 3), device=self.device)
        camera = PerspectiveCameras(
            K=K,
            R=R,
            T=T,
            in_ndc=False,
            image_size=((self.size, self.size),),
            device=self.device,
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
        flow_tensor = rearrange(flow, "(w h b) c -> b c h w", w=self.size, h=self.size)

        importance = 1.0 / (transformed_z)
        importance_min = importance.amin(keepdim=True)
        importance_max = importance.amax(keepdim=True)
        weights = (importance - importance_min) / (
            importance_max - importance_min + 1e-6
        ) * 20 - 10
        weights = rearrange(weights, "(w h b) c -> b c h w", w=self.size, h=self.size)

        transformed_z_tensor = rearrange(
            transformed_z, "(w h b) c -> b c h w", w=self.size, h=self.size
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
        # get 3d points from model.current_points_3d
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
        big_resolution = (self.config["antialiasing_factor"] + 1) * self.size
        border_size = (
            big_resolution - self.config["antialiasing_factor"] * self.size
        ) // 2
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
        h = torch.tensor([self.size], device="cuda")
        w = torch.tensor([self.size], device="cuda")
        K = torch.eye(4)[None].to("cuda")
        K[0, 0, 2] = self.size // 2
        K[0, 1, 2] = self.size // 2
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
    def make_inpaint_condition(self, image, image_mask):
        image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

        assert (
            image.shape[0:1] == image_mask.shape[0:1]
        ), "image and image_mask must have the same image size"

        kernel = np.ones((5, 5), np.uint8)
        image_mask = cv2.dilate(image_mask, kernel, iterations=3)
        image[image_mask > 0.5] = -1.0  # set as masked pixel
        image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return image

    @torch.no_grad()
    def inpaint(self, warped_image, inpaint_mask):
        # Convert tensor images to PIL images for image + mask
        image_pil = ToPILImage()(warped_image[0])
        mask_np = np.array(ToPILImage()(inpaint_mask[0]).convert("L")).astype(
            np.float32
        )

        # Expand the inpainting mask by dilation to improve edge blending
        kernel = np.ones((5, 5), np.uint8)
        mask_dilated = cv2.dilate(mask_np, kernel, iterations=3)
        mask_pil = Image.fromarray(mask_dilated.astype(np.uint8))

        # Run inpainting pipeline call
        result = self.inpainting_pipeline(
            prompt=self.inpainting_prompt,
            negative_prompt=self.config["negative_inpainting_prompt"],
            control_image=image_pil,
            control_mask=mask_pil,
            num_inference_steps=self.config["num_inpainting_steps"],
            guidance_scale=self.classifier_free_guidance_scale,
            num_images_per_prompt=1,
            height=self.config["inpainting_resolution"],
            width=self.config["inpainting_resolution"],
            controlnet_conditioning_scale=1.0,
            callback_on_step_end=self.latent_storer,
            callback_on_step_end_tensor_inputs=["latents"],
        )

        # Get results
        best_index = 0
        inpainted_image = result.images[best_index]
        latent = self.latent_storer.latent[[best_index]]
        inpainted_image = ToTensor()(inpainted_image).unsqueeze(0).to(self.device)
        latent = latent.float()

        return {
            "inpainted_image": inpainted_image,
            "latent": latent,
            "best_index": best_index,
        }

    @torch.no_grad()
    def update_depth(self, inpainted_image):
        new_depth, new_disparity = self.get_depth(inpainted_image)
        self.depths.append(new_depth.detach())
        self.disparities.append(new_disparity.detach())

    def update_images_masks(self, latent, inpaint_mask):
        decoded_image = self.decode_latents(latent).detach()
        self.images.append(decoded_image)
        self.masks.append(inpaint_mask)

    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
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

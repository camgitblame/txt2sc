import kornia
import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio.v3 as iio

# from torchvision.io import write_video


def tensor2im(input_image, imtype=np.uint8):
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = (
            image_tensor[0].clamp(0.0, 1.0).cpu().float().numpy()
        )  # convert it into a numpy array
        image_numpy = (
            np.transpose(image_numpy, (1, 2, 0)) * 255.0
        )  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


class LatentStorer:
    def __init__(self):
        self.latent = None

    def __call__(self, pipeline, step, timestep, tensors):
        # 'tensors' is a dict, e.g. {"latents": tensor}
        self.latent = tensors["latents"].detach().to(pipeline.device)
        return {}  # must return a dict, can update latents etc. if desired


def sobel_filter(disp, mode="sobel", beta=10.0):
    sobel_grad = kornia.filters.spatial_gradient(disp, mode=mode, normalized=False)
    sobel_mag = torch.sqrt(
        sobel_grad[:, :, 0, Ellipsis] ** 2 + sobel_grad[:, :, 1, Ellipsis] ** 2
    )
    alpha = torch.exp(-1.0 * beta * sobel_mag).detach()

    return alpha


def apply_colormap(image, cmap="viridis"):
    colormap = plt.get_cmap(cmap)
    colormap = torch.tensor(colormap.colors).to(image.device)
    image_long = (image * 255).long()
    image_long_min = torch.min(image_long)
    image_long_max = torch.max(image_long)
    assert image_long_min >= 0, f"the min value is {image_long_min}"
    assert image_long_max <= 255, f"the max value is {image_long_max}"
    return colormap[image_long[..., 0]]


def apply_depth_colormap(
    depth,
    near_plane=None,
    far_plane=None,
    cmap="viridis",
):
    near_plane = near_plane or float(torch.min(depth))
    far_plane = far_plane or float(torch.max(depth))

    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = torch.clip(depth, 0, 1)

    colored_image = apply_colormap(depth, cmap=cmap)

    return colored_image


# def save_video(video, path, fps=10):
#     video = video.permute(0, 2, 3, 1)
#     video_codec = "libx264"
#     video_options = {
#         "crf": "23",  # Constant Rate Factor (lower value = higher quality, 18 is a good balance)
#         "preset": "slow",
#     }
#     write_video(
#         str(path), video, fps=fps, video_codec=video_codec, options=video_options
#     )


def save_video(video, path, fps=10):
    video_np = video.permute(0, 2, 3, 1).cpu().numpy()
    # Convert to uint8 if necessary
    if video_np.dtype != "uint8":
        video_np = (video_np * 255).clip(0, 255).astype("uint8")
    iio.imwrite(
        str(path),
        video_np,
        fps=fps,
        codec="libx264",
        quality=8,  # Lower = better quality, larger file. 4-6 is common.
    )

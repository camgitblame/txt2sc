import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image


def image_grid(imgs, rows, cols, resize=256):
    """Create a grid of resized PIL images."""
    if resize is not None:
        imgs = [img.resize((resize, resize)) for img in imgs]

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))

    return grid


# ---------------------
# Model Setup
# ---------------------
device = "cuda"
model_path = "camgitblame/indiana-sd15"

finetuned_sd = StableDiffusionInpaintPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
).to(device)

# ---------------------
# Prompt + Generation
# ---------------------
prompt = (
    "POV, walkthrough, temple of doom from idj indianajones, masterpiece, indoor scene"
)
guidance_scale = 7.5
num_samples = 6
generator = torch.Generator(device=device).manual_seed(0)

empty_mask = Image.open("empty_mask.png").convert("RGB")

images = finetuned_sd(
    prompt=prompt,
    image=empty_mask,
    mask_image=empty_mask,
    guidance_scale=guidance_scale,
    generator=generator,
    num_images_per_prompt=num_samples,
).images

# ---------------------
# Save Grid
# ---------------------
grid = image_grid(images, rows=2, cols=3)
grid.save("indiana15_output_grid.jpg")

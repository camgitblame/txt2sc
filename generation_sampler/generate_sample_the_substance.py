import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image


def image_grid(imgs, rows, cols, resize=None):
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
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
model_path = "camgitblame/substance"

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    model_path,
    torch_dtype=dtype,
).to(device)

# ---------------------
# Prompt + Generation
# ---------------------
prompt = "POV, walkthrough, Elisabeth Sparkle's apartment, The Substance, masterpiece, indoor scene, best quality"

# Five distinct seeds, including 13
seeds = [13, 123, 2024, 777, 0]
generators = [torch.Generator(device=device).manual_seed(s) for s in seeds]

guidance_scale = 7.5
num_images = 5  # one per seed

# Use an all-white mask if you want full generation
empty_mask = Image.open("empty_mask.png").convert("RGB")

images = pipe(
    prompt=prompt,
    image=empty_mask,
    mask_image=empty_mask,
    guidance_scale=guidance_scale,
    generator=generators,  # different seed per image
    num_images_per_prompt=num_images,
).images

# ---------------------
# Save Grid
# ---------------------
grid = image_grid(images, rows=1, cols=5)
grid.save("substance.jpg")

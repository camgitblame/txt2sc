import torch
from diffusers import StableDiffusion3Pipeline
from PIL import Image


def image_grid(imgs, rows, cols, resize=256):
    if resize is not None:
        imgs = [img.resize((resize, resize)) for img in imgs]
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


device = "cuda"
model_path = "indiana/indiana-sd3"

print("Loading model...")
pipe = StableDiffusion3Pipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
).to(device)
print("Model loaded.")

prompt = (
    "POV, walkthrough, temple of doom from idj indianajones, masterpiece, indoor scene"
)
guidance_scale = 7.5
num_samples = 6
seeds = [0, 1, 42, 1234]  # Seed values for reproducibility

for seed in seeds:
    print(f"Generating images for seed {seed}")
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        images = pipe(
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_samples,
            generator=generator,
        ).images
    grid = image_grid(images, rows=2, cols=3)
    filename = f"indiana3_1500_grid_seed{seed}.jpg"
    grid.save(filename)
    print(f"Saved: {filename}")

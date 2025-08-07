from PIL import Image

# Create a white RGB mask of size 512x512
empty_mask = Image.new("RGB", (512, 512), color=(255, 255, 255))

# Save to file
empty_mask.save("empty_mask.png")

print("Saved empty_mask.png (512x512, white RGB)")

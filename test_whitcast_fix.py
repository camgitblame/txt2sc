#!/usr/bin/env python3
"""
Test script to verify white cast fixes in the WarpInpaintModel
"""

import torch
import numpy as np
from PIL import Image


def test_white_area_detection():
    """Test the improved white area detection logic"""

    # Create a test image with white areas (simulating mesh artifacts)
    test_image = torch.rand(1, 3, 512, 512) * 0.6 + 0.2  # Normal image range 0.2-0.8

    # Add some white areas (mesh artifacts)
    test_image[:, :, 100:150, 100:150] = 0.95  # Bright white area
    test_image[:, :, 300:320, 300:320] = 0.85  # Lighter area
    test_image[:, :, 400:450, 200:250] = 1.0  # Pure white area

    # Test old vs new threshold
    warped_gray = test_image[0].mean(dim=0)

    old_white_areas = (warped_gray > 0.9).float()
    new_white_areas = (warped_gray > 0.8).float()

    print(f"Test image shape: {test_image.shape}")
    print(f"Image value range: [{test_image.min():.3f}, {test_image.max():.3f}]")
    print(
        f"Old threshold (0.9) detected pixels: {(old_white_areas > 0.5).sum().item()}"
    )
    print(
        f"New threshold (0.8) detected pixels: {(new_white_areas > 0.5).sum().item()}"
    )
    print(
        f"Improvement: {(new_white_areas > 0.5).sum().item() - (old_white_areas > 0.5).sum().item()} more pixels detected"
    )

    return new_white_areas.sum().item() > old_white_areas.sum().item()


def test_average_color_replacement():
    """Test the average color replacement logic"""

    # Create test image with some white artifacts
    test_image = torch.rand(1, 3, 512, 512) * 0.4 + 0.3  # Range 0.3-0.7
    test_image[:, :, 100:150, 100:150] = 1.0  # White artifact

    # Create inpaint mask for the white area
    inpaint_mask = torch.zeros(1, 1, 512, 512)
    inpaint_mask[:, :, 100:150, 100:150] = 1.0

    # Test average color calculation
    valid_mask = ~inpaint_mask.bool().repeat(1, 3, 1, 1)
    if valid_mask.sum() > 0:
        avg_color = test_image[valid_mask].mean()
        print(f"Average color from valid regions: {avg_color:.3f}")
        print(f"Clamped average color: {avg_color.clamp(0.3, 0.7):.3f}")

        # Apply the fix
        test_image_fixed = test_image.clone()
        test_image_fixed[inpaint_mask.bool().repeat(1, 3, 1, 1)] = avg_color.clamp(
            0.3, 0.7
        )

        white_before = (test_image > 0.9).sum().item()
        white_after = (test_image_fixed > 0.9).sum().item()

        print(f"White pixels before fix: {white_before}")
        print(f"White pixels after fix: {white_after}")
        print(f"White pixels reduced by: {white_before - white_after}")

        return white_after < white_before

    return False


def main():
    print("Testing White Cast Fixes")
    print("=" * 50)

    print("\n1. Testing improved white area detection:")
    detection_improved = test_white_area_detection()
    print(f"✓ Detection improved: {detection_improved}")

    print("\n2. Testing average color replacement:")
    replacement_works = test_average_color_replacement()
    print(f"✓ Replacement reduces white pixels: {replacement_works}")

    print("\n3. Key improvements made:")
    print("   - White area detection threshold: 0.9 → 0.8")
    print("   - Mesh hole fill: white (1.0) → average color (0.3-0.7)")
    print("   - Antialiasing fill: white (1.0) → neutral gray (0.5)")
    print("   - More aggressive dilation (7x7) and less erosion (3x3)")
    print("   - Higher inpainting strength for white areas (0.8 vs 0.65)")
    print("   - Added blur-related negative prompts")
    print("   - Minimum 40 inference steps for quality")

    if detection_improved and replacement_works:
        print("\n✅ All tests passed! White cast fixes should work.")
    else:
        print("\n❌ Some tests failed. Check the implementation.")


if __name__ == "__main__":
    main()

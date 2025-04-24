import os
import cv2
import numpy as np
from tqdm import tqdm

def extract_boundary(mask):
    # Normalize to improve contrast
    contrast_stretched = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX)
    
    # Threshold to ensure binary mask
    _, binary_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    
    # Morphological gradient to extract boundary
    kernel = np.ones((3, 3), np.uint8)
    boundary = cv2.morphologyEx(binary_mask, cv2.MORPH_GRADIENT, kernel)

    # Fallback: use Canny if result too faint
    if np.sum(boundary) < 10:
        boundary = cv2.Canny(contrast_stretched, 50, 150)

    return boundary

def process_masks(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    # List all image files in input folder
    mask_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for filename in tqdm(mask_files, desc="Processing masks"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Load grayscale mask
        mask = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: Could not read {input_path}. Skipping.")
            continue

        # Extract boundary
        boundary_mask = extract_boundary(mask)

        # Save boundary mask
        cv2.imwrite(output_path, boundary_mask)

    print(f"✅ All masks processed and saved to: {output_folder}")

# Example usage
input_dir = "./data/new_data/valid/label/"
output_dir = "./data/new_data/valid/boundary_masks"
process_masks(input_dir, output_dir)

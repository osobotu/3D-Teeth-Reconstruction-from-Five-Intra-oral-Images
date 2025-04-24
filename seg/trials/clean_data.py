import os
import shutil
import cv2
import numpy as np

def mask_valid(mask, image_shape, min_area=500):
    # Check shape match
    if mask.shape[:2] != image_shape[:2]:
        return False

    # Check mask is not blank
    if np.sum(mask) == 0:
        return False

    # Check mask has a meaningful contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    max_area = max(cv2.contourArea(cnt) for cnt in contours)
    if max_area < min_area:
        return False

    return True

def clean_and_rename(folder_path):
    files = sorted(os.listdir(folder_path))
    image_mask_pairs = {}

    # Group images and their corresponding masks
    for file in files:
        if '_mask' in file:
            base = file.replace('_mask.png', '')
            image_mask_pairs.setdefault(base, {})['mask'] = file
        else:
            base = os.path.splitext(file)[0]
            image_mask_pairs.setdefault(base, {})['image'] = file

    count = 0
    skipped = 0
    for base, pair in sorted(image_mask_pairs.items()):
        if 'image' not in pair or 'mask' not in pair:
            print(f"⚠️ Skipping incomplete pair: {pair}")
            skipped += 1
            continue

        # Load image and mask
        image_path = os.path.join(folder_path, pair['image'])
        mask_path = os.path.join(folder_path, pair['mask'])
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            print(f"⚠️ Could not load image or mask for: {base}")
            skipped += 1
            continue

        # if not mask_valid(mask, image.shape):
        #     print(f"❌ Invalid mask for: {pair['mask']} — Skipping.")
        #     skipped += 1
        #     continue

        # Generate new names
        new_image_name = f"image_{count}.jpg"
        new_mask_name = f"image_{count}_mask.png"

        # Paths for renamed files
        new_image_path = os.path.join(folder_path, new_image_name)
        new_mask_path = os.path.join(folder_path, new_mask_name)

        # Rename files
        os.rename(image_path, new_image_path)
        os.rename(mask_path, new_mask_path)
        count += 1

    print(f"\n✅ Renamed {count} valid image-mask pairs.")
    print(f"🚫 Skipped {skipped} invalid or incomplete pairs.")


folder = "./data/new_data/valid"
clean_and_rename(folder)

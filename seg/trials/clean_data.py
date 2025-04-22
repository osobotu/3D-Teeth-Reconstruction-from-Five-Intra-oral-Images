import os
import shutil

def clean_and_rename(folder_path):
    files = sorted(os.listdir(folder_path))  # sort for consistency
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
    for _, pair in sorted(image_mask_pairs.items()):
        if 'image' not in pair or 'mask' not in pair:
            print(f"Skipping incomplete pair: {pair}")
            continue

        # Generate new names
        new_image_name = f"image_{count}.jpg"
        new_mask_name = f"image_{count}_mask.png"

        # Paths
        old_image_path = os.path.join(folder_path, pair['image'])
        old_mask_path = os.path.join(folder_path, pair['mask'])
        new_image_path = os.path.join(folder_path, new_image_name)
        new_mask_path = os.path.join(folder_path, new_mask_name)

        # Rename files
        os.rename(old_image_path, new_image_path)
        os.rename(old_mask_path, new_mask_path)

        count += 1

    print(f"Renamed {count} image-mask pairs successfully.")


folder = "./data/valid"
clean_and_rename(folder)

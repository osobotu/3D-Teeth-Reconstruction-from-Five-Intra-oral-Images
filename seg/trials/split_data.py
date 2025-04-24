import os
import shutil

def split_images_and_labels(source_folder):
    images_dir = os.path.join(source_folder, "image")
    labels_dir = os.path.join(source_folder, "label")

    # Create folders if they don't exist
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    for filename in os.listdir(source_folder):
        filepath = os.path.join(source_folder, filename)

        # Skip directories
        if os.path.isdir(filepath):
            continue

        if '_mask' in filename.lower():
            shutil.move(filepath, os.path.join(labels_dir, filename))
        elif filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            shutil.move(filepath, os.path.join(images_dir, filename))

    print(f"Split complete: images → {images_dir}, labels → {labels_dir}")

# Example usage
source = "./data/new_data/valid"
split_images_and_labels(source)

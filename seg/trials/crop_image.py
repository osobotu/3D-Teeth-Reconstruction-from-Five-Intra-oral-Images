import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def crop_to_mask_region(image, mask, padding=10):
    """
    Crop the original image to only include regions where the mask has content,
    with optional padding around that region.
    """
    # Find non-zero points in the mask
    non_zero_points = np.where(mask > 0)
    
    # If mask is empty, return the original image
    if len(non_zero_points[0]) == 0:
        return image
    
    # Get bounds of non-zero points
    min_y, max_y = np.min(non_zero_points[0]), np.max(non_zero_points[0])
    min_x, max_x = np.min(non_zero_points[1]), np.max(non_zero_points[1])
    
    # Add padding
    min_y = max(0, min_y - padding)
    max_y = min(image.shape[0], max_y + padding)
    min_x = max(0, min_x - padding)
    max_x = min(image.shape[1], max_x + padding)
    
    # Crop image and mask
    cropped_image = image[min_y:max_y, min_x:max_x]
    cropped_mask = mask[min_y:max_y, min_x:max_x]
    
    return cropped_image, cropped_mask, (min_y, max_y, min_x, max_x)

def process_dataset(images_folder, masks_folder, output_images_folder, output_masks_folder, padding=10, visualize_samples=5):
    """
    Process all image-mask pairs by cropping images to only include mask regions.
    """
    os.makedirs(output_images_folder, exist_ok=True)
    os.makedirs(output_masks_folder, exist_ok=True)
    
    # Create visualization folder if needed
    viz_folder = os.path.join(output_images_folder, "visualization")
    if visualize_samples > 0:
        os.makedirs(viz_folder, exist_ok=True)
    
    # Get all mask files
    mask_files = [f for f in os.listdir(masks_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Select random files for visualization
    if visualize_samples > 0:
        viz_files = np.random.choice(mask_files, min(visualize_samples, len(mask_files)), replace=False)
    else:
        viz_files = []
    
    # Stats for summary
    total_processed = 0
    empty_masks = 0
    
    for filename in tqdm(mask_files, desc="Processing images"):
        mask_path = os.path.join(masks_folder, filename)
        
        # Construct corresponding image filename
        # Adjust this logic based on your naming convention
        base_name = os.path.splitext(filename)[0]
        image_filename = f"{base_name.replace('_mask', '')}.png"  # Adjust as needed
        image_path = os.path.join(images_folder, image_filename)
        
        # Check if image exists
        if not os.path.exists(image_path):
            # Try alternative extensions
            for ext in ['.jpg', '.jpeg', '.tif', '.tiff']:
                alt_path = os.path.join(images_folder, f"{base_name.replace('_mask', '')}{ext}")
                if os.path.exists(alt_path):
                    image_path = alt_path
                    break
            
            if not os.path.exists(image_path):
                print(f"Warning: Could not find image for mask {filename}. Skipping.")
                continue
        
        # Load image and mask
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None or mask is None:
            print(f"Warning: Could not read image or mask for {filename}. Skipping.")
            continue
        
        # Check if mask has any content
        if np.sum(mask) == 0:
            print(f"Warning: Mask for {filename} is empty. Skipping.")
            empty_masks += 1
            continue
        
        # Crop image and mask to region of interest
        cropped_image, cropped_mask, crop_coords = crop_to_mask_region(image, mask, padding)
        
        # Save cropped image and mask
        output_image_path = os.path.join(output_images_folder, image_filename)
        output_mask_path = os.path.join(output_masks_folder, filename)
        
        cv2.imwrite(output_image_path, cropped_image)
        cv2.imwrite(output_mask_path, cropped_mask)
        
        total_processed += 1
        
        # Create visualization for sample images
        if filename in viz_files:
            plt.figure(figsize=(15, 10))
            
            # Original image and mask
            plt.subplot(2, 2, 1)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title("Original Image")
            plt.axis('off')
            
            plt.subplot(2, 2, 2)
            plt.imshow(mask, cmap='gray')
            plt.title("Original Mask")
            plt.axis('off')
            
            # Cropped image and mask
            plt.subplot(2, 2, 3)
            plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
            plt.title(f"Cropped Image (coords: {crop_coords})")
            plt.axis('off')
            
            plt.subplot(2, 2, 4)
            plt.imshow(cropped_mask, cmap='gray')
            plt.title("Cropped Mask")
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(viz_folder, f"viz_{filename}"))
            plt.close()
    
    print(f"✅ Processed {total_processed} images")
    print(f"⚠️ Found {empty_masks} empty masks")
    print(f"Output saved to: {output_images_folder} and {output_masks_folder}")

def main():
    # Set your directories
    images_folder = "./data/new_data/train/image"
    masks_folder = "./data/new_data/train/label"
    output_images_folder = "./data/new_data/train/cropped_images"
    output_masks_folder = "./data/new_data/train/cropped_masks"
    
    # Process dataset
    process_dataset(
        images_folder, 
        masks_folder, 
        output_images_folder, 
        output_masks_folder, 
        padding=20,  # Adjust padding as needed
        visualize_samples=5
    )

if __name__ == "__main__":
    main()
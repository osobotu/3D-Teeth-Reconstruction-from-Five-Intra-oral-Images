# import os
# import cv2
# import numpy as np
# from tqdm import tqdm

# def extract_boundary(mask):
#     # Normalize to improve contrast
#     contrast_stretched = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX)
    
#     # Threshold to ensure binary mask
#     _, binary_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    
#     # Morphological gradient to extract boundary
#     kernel = np.ones((3, 3), np.uint8)
#     boundary = cv2.morphologyEx(binary_mask, cv2.MORPH_GRADIENT, kernel)

#     # Fallback: use Canny if result too faint
#     if np.sum(boundary) < 10:
#         boundary = cv2.Canny(contrast_stretched, 50, 150)

#     return boundary

# def process_masks(input_folder, output_folder):
#     os.makedirs(output_folder, exist_ok=True)
    
#     # List all image files in input folder
#     mask_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

#     for filename in tqdm(mask_files, desc="Processing masks"):
#         input_path = os.path.join(input_folder, filename)
#         output_path = os.path.join(output_folder, filename)

#         # Load grayscale mask
#         mask = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
#         if mask is None:
#             print(f"Warning: Could not read {input_path}. Skipping.")
#             continue

#         # Extract boundary
#         boundary_mask = extract_boundary(mask)

#         # Save boundary mask
#         cv2.imwrite(output_path, boundary_mask)

#     print(f"✅ All masks processed and saved to: {output_folder}")

# # Example usage
# input_dir = "./data/new_data/valid/label/"
# output_dir = "./data/new_data/valid/boundary_masks"
# process_masks(input_dir, output_dir)

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def add_padding(img, pad=20):
    """Add padding around the image to prevent cropping during transformations"""
    return cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

def center_mask_contour(mask):
    """Center the mask using contour moments for more precise centering"""
    # Find contours
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, 
                                  cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return mask
    
    # Find the largest contour (should be the teeth)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get the center of the contour
    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        return mask
    
    # Calculate translation to center
    h, w = mask.shape[:2]
    dx = w//2 - cX
    dy = h//2 - cY
    
    # Apply translation
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    centered = cv2.warpAffine(mask, M, (w, h))
    
    return centered

def extract_boundary_contour(mask):
    """Extract boundary using contour detection for smoother results"""
    # Ensure mask is binary and has proper intensity values
    _, binary_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # Create empty boundary image
    boundary = np.zeros_like(binary_mask)
    
    # Draw contours
    cv2.drawContours(boundary, contours, -1, 255, 1)
    
    # If no contours found or result too faint, try internal contours
    if np.sum(boundary) < 100:
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(boundary, contours, -1, 255, 1)
    
    # Final fallback: use Canny edge detection
    if np.sum(boundary) < 100:
        contrast_stretched = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX)
        boundary = cv2.Canny(contrast_stretched, 30, 150)
    
    return boundary

def process_masks(input_folder, output_folder, visualize_samples=5):
    """Process all masks in input folder and save boundaries to output folder"""
    os.makedirs(output_folder, exist_ok=True)
    
    # Create visualization folder if needed
    viz_folder = os.path.join(output_folder, "visualization")
    if visualize_samples > 0:
        os.makedirs(viz_folder, exist_ok=True)
    
    # List all image files in input folder
    mask_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Select random files for visualization
    if visualize_samples > 0:
        viz_files = np.random.choice(mask_files, min(visualize_samples, len(mask_files)), replace=False)
    else:
        viz_files = []
    
    for filename in tqdm(mask_files, desc="Processing masks"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        # Load grayscale mask
        mask = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: Could not read {input_path}. Skipping.")
            continue
        
        # Log unique values and shape for debugging
        if filename in viz_files:
            print(f"Mask {filename}: shape={mask.shape}, unique values={np.unique(mask)}")
        
        # Add padding to prevent cropping
        padded_mask = add_padding(mask, pad=30)
        
        # Center the mask
        centered_mask = center_mask_contour(padded_mask)
        
        # Extract boundary
        boundary_mask = extract_boundary_contour(centered_mask)
        
        # Save boundary mask
        cv2.imwrite(output_path, boundary_mask)
        
        # Create visualization for sample images
        if filename in viz_files:
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 4, 1)
            plt.imshow(mask, cmap='gray')
            plt.title("Original Mask")
            plt.axis('off')
            
            plt.subplot(1, 4, 2)
            plt.imshow(padded_mask, cmap='gray')
            plt.title("Padded Mask")
            plt.axis('off')
            
            plt.subplot(1, 4, 3)
            plt.imshow(centered_mask, cmap='gray')
            plt.title("Centered Mask")
            plt.axis('off')
            
            plt.subplot(1, 4, 4)
            plt.imshow(boundary_mask, cmap='gray')
            plt.title("Boundary Mask")
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(viz_folder, f"viz_{filename}"))
            plt.close()
    
    print(f"✅ All masks processed and saved to: {output_folder}")

def visualize_single_mask(mask_path):
    """Visualize processing steps for a single mask"""
    # Load the grayscale mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error: Could not read {mask_path}")
        return
    
    print(f"Mask shape: {mask.shape}")
    print(f"Unique values: {np.unique(mask)}")
    
    # Add padding
    padded_mask = add_padding(mask, pad=30)
    
    # Center the mask
    centered_mask = center_mask_contour(padded_mask)
    
    # Binarize the mask
    _, binary_mask = cv2.threshold(centered_mask, 1, 255, cv2.THRESH_BINARY)
    
    # Extract boundary
    boundary_mask = extract_boundary_contour(centered_mask)
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 4, 1)
    plt.imshow(mask, cmap='gray')
    plt.title("Original Mask")
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.imshow(centered_mask, cmap='gray')
    plt.title("Centered Mask")
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.imshow(binary_mask, cmap='gray')
    plt.title("Binary Mask")
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.imshow(boundary_mask, cmap='gray')
    plt.title("Boundary Mask")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return boundary_mask

def main():
    # Set input and output directories
    input_dir = "./data/new_data/valid/label/"
    output_dir = "./data/new_data/valid/boundary_masks"
    
    # Process all masks
    # process_masks(input_dir, output_dir, visualize_samples=5)
    
    # Or process a single mask for testing
    test_mask_path = './IMG_2473_jpeg.rf.a3877c8222c3b9f78d9e7980e877285b_mask.png'
    visualize_single_mask(test_mask_path)

if __name__ == "__main__":
    main()

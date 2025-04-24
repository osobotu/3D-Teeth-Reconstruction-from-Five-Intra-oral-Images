import os
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.segmentation import active_contour
from matplotlib import pyplot as plt
from tqdm import tqdm

def detect_mismatched_boundaries(image_folder, mask_folder, output_folder, threshold=0.5):
    """
    Detects mismatched teeth boundary masks by comparing edge features.
    
    Args:
        image_folder: Folder containing original teeth images
        mask_folder: Folder containing boundary masks
        output_folder: Folder to save mismatched pairs and reports
        threshold: Similarity threshold below which pairs are considered mismatched
    
    Returns:
        List of mismatched image filenames
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get lists of files in both folders
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    mask_files = sorted([f for f in os.listdir(mask_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    # Check if we have matching file counts
    if len(image_files) != len(mask_files):
        print(f"Warning: Number of images ({len(image_files)}) doesn't match number of masks ({len(mask_files)})")
    
    # Function to extract edges from teeth image
    def extract_edges(img):
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding to handle varying lighting
        thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Extract edges using Canny
        edges = cv2.Canny(thresh, 50, 150)
        return edges
    
    # Function to evaluate mask alignment with image
    def evaluate_alignment(img_edges, mask):
        # Ensure mask is binary
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        # Threshold to ensure binary
        _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Dilate the mask slightly to increase overlap with edges
        kernel = np.ones((3,3), np.uint8)
        mask_dilated = cv2.dilate(mask_binary, kernel, iterations=1)
        
        # Calculate intersection of edges and mask
        intersection = cv2.bitwise_and(img_edges, mask_dilated)
        
        # Calculate IoU (Intersection over Union)
        union = cv2.bitwise_or(img_edges, mask_dilated)
        iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
        
        # Calculate SSIM between edges and mask
        edge_similarity = ssim(img_edges, mask_dilated, data_range=255)
        
        # Calculate feature distance using cosine similarity
        img_edges_flat = img_edges.flatten().astype(float)
        mask_flat = mask_dilated.flatten().astype(float)
        
        img_norm = np.linalg.norm(img_edges_flat)
        mask_norm = np.linalg.norm(mask_flat)
        
        if img_norm > 0 and mask_norm > 0:
            cosine_sim = np.dot(img_edges_flat, mask_flat) / (img_norm * mask_norm)
        else:
            cosine_sim = 0
            
        # Combined score (weighted average of metrics)
        combined_score = (0.4 * iou) + (0.4 * edge_similarity) + (0.2 * cosine_sim)
        
        return {
            "iou": iou,
            "ssim": edge_similarity,
            "cosine_similarity": cosine_sim,
            "combined_score": combined_score
        }
    
    # Process each pair of files
    mismatched_pairs = []
    results = []
    
    print(f"Processing {len(image_files)} image-mask pairs...")
    for i, (img_file, mask_file) in enumerate(tqdm(zip(image_files, mask_files))):
        # Read images
        img_path = os.path.join(image_folder, img_file)
        mask_path = os.path.join(mask_folder, mask_file)
        
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path)
        
        if img is None or mask is None:
            print(f"Error loading {img_path} or {mask_path}")
            continue
        
        # Make sure images are the same size
        if img.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        
        # Extract edges from teeth image
        img_edges = extract_edges(img)
        
        # Evaluate alignment
        metrics = evaluate_alignment(img_edges, mask)
        
        # Save result
        result = {
            "image_file": img_file,
            "mask_file": mask_file,
            "metrics": metrics,
            "is_mismatched": metrics["combined_score"] < threshold
        }
        results.append(result)
        
        # If mismatched, add to list and save visualization
        if result["is_mismatched"]:
            mismatched_pairs.append((img_file, mask_file))
            
            # Create visualization of the mismatch
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax[0].set_title("Original Image")
            
            ax[1].imshow(mask, cmap='gray')
            ax[1].set_title("Boundary Mask")
            
            # Overlay visualization
            overlay = img.copy()
            if len(mask.shape) == 2:
                mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            else:
                mask_colored = mask.copy()
                
            # Make mask red for visualization
            mask_colored[:,:,0] = 0  # Zero out blue channel
            mask_colored[:,:,1] = 0  # Zero out green channel
            
            # Blend images
            overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
            ax[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            ax[2].set_title(f"Overlay (Score: {metrics['combined_score']:.3f})")
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, f"mismatch_{img_file}"))
            plt.close()
    
    # Save summary report
    with open(os.path.join(output_folder, "mismatch_report.txt"), "w") as f:
        f.write(f"Total image-mask pairs analyzed: {len(results)}\n")
        f.write(f"Total mismatched pairs found: {len(mismatched_pairs)}\n\n")
        
        f.write("Mismatched Pairs:\n")
        for img_file, mask_file in mismatched_pairs:
            f.write(f"- {img_file} / {mask_file}\n")
        
        f.write("\n\nDetailed Results:\n")
        for result in results:
            f.write(f"\nImage: {result['image_file']}\n")
            f.write(f"Mask: {result['mask_file']}\n")
            f.write(f"IoU: {result['metrics']['iou']:.4f}\n")
            f.write(f"SSIM: {result['metrics']['ssim']:.4f}\n")
            f.write(f"Cosine Similarity: {result['metrics']['cosine_similarity']:.4f}\n")
            f.write(f"Combined Score: {result['metrics']['combined_score']:.4f}\n")
            f.write(f"Mismatched: {'Yes' if result['is_mismatched'] else 'No'}\n")
    
    return mismatched_pairs

# Example usage
if __name__ == "__main__":
    # Replace these with your actual folder paths
    image_folder = "./data/new_data/train/image"  
    mask_folder = "./data/new_data/train/label"
    output_folder = "./data/new_data/train/result"
    
    # Set threshold - adjust based on your data
    similarity_threshold = 0.4  # Lower values are more permissive
    
    # Detect mismatched pairs
    mismatched = detect_mismatched_boundaries(
        image_folder, 
        mask_folder, 
        output_folder, 
        threshold=similarity_threshold
    )
    
    print(f"Found {len(mismatched)} mismatched image-boundary pairs")
    print(f"Results saved to {output_folder}")
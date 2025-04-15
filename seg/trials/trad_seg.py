import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_teeth_boundaries(input_path, output_path):
    # 1. Load image (convert to grayscale)
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    
    # 2. Preprocessing
    # Denoising
    img = cv2.medianBlur(img, 5)
    
    # Contrast enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    
    # 3. Thresholding (Otsu's method for automatic threshold)
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 4. Morphological operations
    kernel = np.ones((3,3), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # 5. Edge detection (Canny)
    edges = cv2.Canny(closed, 100, 200)
    
    # 6. Find contours (only external contours)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 7. Create output mask
    output = np.zeros_like(img)
    cv2.drawContours(output, contours, -1, (255), 1)  # 1-pixel wide boundaries
    
    # 8. Save result
    cv2.imwrite(output_path, output)
    print(f"Boundaries saved to {output_path}")

    # (Optional) Display results
    plt.figure(figsize=(12,6))
    plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original')
    plt.subplot(122), plt.imshow(output, cmap='gray'), plt.title('Detected Boundaries')
    plt.show()

# Usage example
detect_teeth_boundaries("0-2.png", "0-2-label.png")

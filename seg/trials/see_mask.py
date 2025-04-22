# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # Load the mask
# mask = cv2.imread('test_mask.png', cv2.IMREAD_GRAYSCALE)

# # Print unique pixel values
# unique_vals = np.unique(mask)
# print("Unique pixel values in mask:", unique_vals)

# # Visualize with matplotlib (auto contrast)
# plt.imshow(mask, cmap='gray')
# plt.title("Mask Visualization")
# plt.axis('off')
# plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the grayscale mask
mask_path = './data/train/label/image_82_mask.png'
# mask_path = 'test_mask.png'
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# Print unique pixel values to understand intensity distribution
unique_vals = np.unique(mask)
print("Unique pixel values in mask:", unique_vals)

# Stretch contrast if necessary (for display only)
contrast_stretched = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX)

# Step 1: Try to binarize the mask based on dynamic threshold
_, binary_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

# Step 2: Try extracting boundaries using morphological gradient
kernel = np.ones((3, 3), np.uint8)
gradient = cv2.morphologyEx(binary_mask, cv2.MORPH_GRADIENT, kernel)

# Step 3 (fallback): Use Canny edge detection if gradient is too faint
if np.sum(gradient) < 10:
    print("Fallback to Canny edge detection")
    gradient = cv2.Canny(contrast_stretched, 50, 150)

# Plot results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(mask, cmap='gray')
plt.title("Original Mask")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(binary_mask, cmap='gray')
plt.title("Binary Mask")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(gradient, cmap='gray')
plt.title("Boundary Mask")
plt.axis('off')

plt.tight_layout()
plt.show()



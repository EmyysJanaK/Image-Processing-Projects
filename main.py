import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load a color image and convert to grayscale
img = cv2.imread('input_image.jpg', cv2.IMREAD_GRAYSCALE)

# Create a 3x2 subplot
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# Operations on the image

# get Unprocessed grayscale image
axs[0, 0].imshow(img, cmap='gray')
axs[0, 0].set_title('Original Grayscale')

# get Negative image
negative = 255 - img
axs[0, 1].imshow(negative, cmap='gray')
axs[0, 1].set_title('Negative')

# get Increased brightness by 20%
brightness = np.clip(img * 1.2, 0, 255).astype(np.uint8)
axs[0, 2].imshow(brightness, cmap='gray')
axs[0, 2].set_title('Brightness +20%')

# get Reduce contrast (125-175)
contrast = np.clip(((img - img.min()) / (img.max() - img.min()) * 50 + 125), 0, 255).astype(np.uint8)
axs[1, 0].imshow(contrast, cmap='gray')
axs[1, 0].set_title('Reduced Contrast')

# get Reduce to 4bpp
bpp4 = (img // 16) * 16
axs[1, 1].imshow(bpp4, cmap='gray')
axs[1, 1].set_title('4bpp')

# get Vertical mirror
mirror = np.flip(img, axis=1)
axs[1, 2].imshow(mirror, cmap='gray')
axs[1, 2].set_title('Vertical Mirror')

# Remove axis ticks
for ax in axs.flat:
    ax.axis('off')

# Adjust layout and display
plt.tight_layout()
plt.show()

# Save the output image
cv2.imwrite('output_image.jpg', np.hstack((img, negative, brightness, contrast, bpp4, mirror)))
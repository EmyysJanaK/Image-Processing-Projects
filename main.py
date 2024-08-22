import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load a color image
img = cv2.imread('210598B_SrcImage.jpg')

# 2. Converting the 3D Numpy Array containing BGR Image to a 2D array containing the Grayscale Image
gray_img = np.round(0.114 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.299 * img[:, :, 2]).astype(np.uint8)

# 3x2 subplot
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# Point Operations

# 1. Unprocessed grayscale image
axs[0, 0].imshow(gray_img, cmap='gray')
axs[0, 0].set_title('Original Grayscale')

# 2. Negative image
negative = 255 - gray_img  # Dark pixels -> Bright pixels, Bright pixels -> Dark pixels
axs[0, 1].imshow(negative, cmap='gray')
axs[0, 1].set_title('Negative')

# 3. Increasing img brightness by 20%
brightness = np.clip(gray_img * 1.2, 0, 255).astype(np.uint8)
axs[0, 2].imshow(brightness, cmap='gray')
axs[0, 2].set_title('Brightness +20%')

# 4. Reducing contrast (125-175)
contrast = np.clip(((gray_img - gray_img.min()) / (gray_img.max() - gray_img.min()) * 50 + 125), 0, 255).astype(np.uint8)
axs[1, 0].imshow(contrast, cmap='gray')
axs[1, 0].set_title('Reduced Contrast')

# 5. Reducing to 4bpp
bpp4 = (gray_img // 16) * 16  # 4bpp = 16 shades of gray
axs[1, 1].imshow(bpp4, cmap='gray')
axs[1, 1].set_title('4bpp')

# 6. Vertical mirror
mirror = np.flip(gray_img, axis=1)
axs[1, 2].imshow(mirror, cmap='gray')
axs[1, 2].set_title('Vertical Mirror')

# Remove axis ticks
for ax in axs.flat:
    ax.axis('off')

# Final layout and display
plt.tight_layout()
plt.show()

# Saving the images
cv2.imwrite('210598B_OPImage.jpg', np.hstack((gray_img, negative, brightness, contrast, bpp4, mirror)))
cv2.imwrite('210598B_OPImage_11.jpg', gray_img)
cv2.imwrite('210598B_OPImage_12.jpg', negative)
cv2.imwrite('210598B_OPImage_13.jpg', brightness)
cv2.imwrite('210598B_OPImage_14.jpg', contrast)
cv2.imwrite('210598B_OPImage_15.jpg', bpp4)
cv2.imwrite('210598B_OPImage_16.jpg', mirror)

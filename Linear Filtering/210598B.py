import cv2
import numpy as np

image_path = r"road98.png"
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Could not load image at {image_path}")
else:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(image_rgb.shape)

R = image_rgb[:, :, 0]
G = image_rgb[:, :, 1]
B = image_rgb[:, :, 2]

grayscale_image = (0.299 * R + 0.587 * G + 0.114 * B).astype(np.uint8)

min = np.min(grayscale_image)
max = np.max(grayscale_image)

constrast_enhanced_image = ((grayscale_image - min) / (max - min) * 255).astype(np.uint8)

cv2.imwrite("original.jpg", constrast_enhanced_image)

filter_A = np.array([[0,-1,-1,-1,0],[-1,2,2,2,-1],[-1,2,8,2,-1],[-1,2,2,2,-1],[0,-1,-1,-1,0]])
filter_B = np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]])
filter_C = np.array([[5,5,5,5,5],[5,5,5,5,5],[5,5,5,5,5],[5,5,5,5,5],[5,5,5,5,5]])
filter_D = np.array([[0,-1,-1,-1,0],[-1,2,2,2,-1],[-1,2,16,2,-1],[-1,2,2,2,-1],[0,-1,-1,-1,0]])


def normalize_filter(filter):
    filter_sum = np.sum(filter)
    if filter_sum != 0:
        filter = filter / filter_sum
    return filter

filter_A = normalize_filter(filter_A)
filter_B = normalize_filter(filter_B)
filter_C = normalize_filter(filter_C)
filter_D = normalize_filter(filter_D)

def apply_filter(image, filter_matrix, filter_name):
    image_height, image_width = image.shape
    filter_size = filter_matrix.shape[0]
    
    padding_size = filter_size // 2
    
    padded_image = np.pad(image, padding_size, mode='constant', constant_values=0)
    
    output_image = np.zeros_like(image)
    
    for i in range(image_height):
        for j in range(image_width):
            region = padded_image[i:i+filter_size, j:j+filter_size]
            
            output_value = np.sum(region * filter_matrix)
            
            output_image[i, j] = output_value
    
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)
    
    output_filename = f"{filter_name}.jpg"
    cv2.imwrite(output_filename, output_image)
    
    return output_image

def compute_rms_difference(image1, image2):
    array1 = np.array(image1)
    array2 = np.array(image2)
    diff = np.sqrt(np.mean((array1 - array2) ** 2))
    return diff


filtered_image_A = apply_filter(constrast_enhanced_image, filter_A, "filter_A")
filtered_image_B = apply_filter(constrast_enhanced_image, filter_B, "filter_B")
filtered_image_C = apply_filter(constrast_enhanced_image, filter_C, "filter_C")
filtered_image_D = apply_filter(constrast_enhanced_image, filter_D, "filter_D")

rms_A = compute_rms_difference(constrast_enhanced_image, filtered_image_A)
rms_B = compute_rms_difference(constrast_enhanced_image, filtered_image_B)
rms_C = compute_rms_difference(constrast_enhanced_image, filtered_image_C)
rms_D = compute_rms_difference(constrast_enhanced_image, filtered_image_D)

print(f"RMS Difference for Filter A: {rms_A}")
print(f"RMS Difference for Filter B: {rms_B}")
print(f"RMS Difference for Filter C: {rms_C}")
print(f"RMS Difference for Filter D: {rms_D}")

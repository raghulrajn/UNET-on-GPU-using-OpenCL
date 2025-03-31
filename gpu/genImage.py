#Code to generate sample image for inference and converting them to .npy file for CPU/GPU input
import numpy as np
import random
import cv2
import torch
def preprocess_input(x):
    x = x.astype(np.float64)
    x /= 256.0
    x -= 0.5
    return x

def saveasnpy(image, filename):
  input_data = cv2.imread(image)  # Shape: (H, W, C)
  input_data = preprocess_input(input_data)
  # Reshape to NCHW: (1, 3, 224, 224)
  input_data = input_data.transpose(2, 0, 1)  # From (H, W, C) to (C, H, W)
  input_data = input_data[np.newaxis, ...]    # Add batch dimension: (1, C, H, W)
  input_data = input_data.astype(np.float32)  # Ensure float32 for PyTorch
  image_tensor = torch.from_numpy(input_data) # Shape: (1, 3, 224, 224)
  np.save(filename, image_tensor)

def gen_random_image():
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    mask = np.zeros((224, 224), dtype=np.uint8)

    # Background
    dark_color0 = random.randint(0, 100)
    dark_color1 = random.randint(0, 100)
    dark_color2 = random.randint(0, 100)
    img[:, :, 0] = dark_color0
    img[:, :, 1] = dark_color1
    img[:, :, 2] = dark_color2

    # Object
    light_color0 = random.randint(dark_color0+1, 255)
    light_color1 = random.randint(dark_color1+1, 255)
    light_color2 = random.randint(dark_color2+1, 255)
    center_0 = random.randint(0, 224)
    center_1 = random.randint(0, 224)
    r1 = random.randint(10, 56)
    r2 = random.randint(10, 56)
    cv2.ellipse(img, (center_0, center_1), (r1, r2), 0, 0, 360, (light_color0, light_color1, light_color2), -1)
    cv2.ellipse(mask, (center_0, center_1), (r1, r2), 0, 0, 360, 255, -1)

    # White noise
    density = random.uniform(0, 0.1)
    for i in range(224):
        for j in range(224):
            if random.random() < density:
                img[i, j, 0] = random.randint(0, 255)
                img[i, j, 1] = random.randint(0, 255)
                img[i, j, 2] = random.randint(0, 255)

    return img, mask

img, mask = gen_random_image()

cv2.imwrite("img4.png", img)
saveasnpy("img4.png", "img4.npy")

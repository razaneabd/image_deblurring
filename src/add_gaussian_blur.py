import cv2
import os
import random
import numpy as np
from tqdm import tqdm


def add_gaussian_noise(image):
    """Add Gaussian noise to the image"""
    noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
    return cv2.add(image, noise)

os.makedirs('./input/gaussian_blurred', exist_ok = True)
src_dir = './input/sharp'
images = os.listdir(src_dir)
dst_dir = './input/gaussian_blurred'

for i, img in tqdm(enumerate(images), total = len(images)):
    img = cv2.imread(f"{src_dir}/{images[i]}", cv2.IMREAD_COLOR)
    kernel_size = random.choice([15, 17, 21, 27, 31, 35])  
    blur = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    #blur = cv2.GaussianBlur(img, (31, 31), 0) # Kernel size (31, 31)
    cv2.imwrite(f"{dst_dir}/{images[i]}", blur)
print("Completed Blurring!")
import cv2
import os
import random
import numpy as np
from tqdm import tqdm

os.makedirs('./input/gaussian_blurred', exist_ok = True)
src_dir = './input/sharp'
images = os.listdir(src_dir)
dst_dir = './input/gaussian_blurred'

# Define augmentation functions
for i, img_name in tqdm(enumerate(images), total=len(images)):
    img_path = os.path.join(src_dir, img_name)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    if img is None:
        continue  # Skip invalid images

    # Apply random rotation 
    angle = random.uniform(-15, 15)
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    augmented_img = cv2.warpAffine(img, M, (w, h))

    # Random brightness adjustment 
    value = random.randint(-10, 10)
    hsv = cv2.cvtColor(augmented_img, cv2.COLOR_BGR2HSV)
    hsv[..., 2] = np.clip(hsv[..., 2] + value, 0, 255)
    augmented_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Save the final augmented and blurred image with "_aug" appended to the filename
    new_img_name = img_name.split('.')[0] + '_aug.' + img_name.split('.')[1]
    cv2.imwrite(os.path.join(src_dir, new_img_name), augmented_img)

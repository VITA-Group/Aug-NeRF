import random
import cv2
import numpy as np
import os
 
def add_noise(img, noise_ratio):
 
    # Getting the dimensions of the image
    row , col, _ = img.shape
    n_noise = int(row * col * noise_ratio / 2)

    coords = np.random.randint((0, 0), (row, col), (n_noise * 2, 2))
    y_coord, x_coord = coords[..., 0], coords[..., 1]
    img[y_coord[:n_noise], x_coord[:n_noise], :] = 0
    img[y_coord[n_noise:], x_coord[n_noise:], :] = 255
         
    return img

input_dir = sys.argv[1]
output_dir = sys.argv[2]
ratio = 0.3
os.makedirs(output_dir, exist_ok=True)

for fn in os.listdir(input_dir):
    if os.path.splitext(fn)[1].lower() not in ['.jpg', '.png']:
        continue
    img_path = os.path.join(input_dir, fn)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    proc_img = add_noise(img, noise_ratio=ratio)
    cv2.imwrite(os.path.join(output_dir, fn), proc_img)
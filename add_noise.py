from audioop import add
import os
import cv2
import numpy as np
import random

#two kinds of noises (salt noise and gaussian noise)
source_path = "source"
salt_output_path = "salt"
gaussian_output_path = "gaussian"

salt_prob = 0.02

gaussian_sigma = 0.08

def add_salt_noise():
    files = os.listdir(source_path)
    for idx, f in enumerate(files):
        p1 = os.path.join(source_path, f)
        p2 = os.path.join(salt_output_path, f)
        img = cv2.imread(p1, cv2.IMREAD_GRAYSCALE)
        res = np.zeros(img.shape, np.uint8)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                r = random.random()
                if r < salt_prob:
                    res[i][j] = 0
                elif r > 1 - salt_prob:
                    res[i][j] = 255
                else:
                    res[i][j] = img[i][j]
        cv2.imwrite(p2, res)

def add_gaussian_noise():
    files = os.listdir(source_path)
    for idx, f in enumerate(files):
        p1 = os.path.join(source_path, f)
        p2 = os.path.join(gaussian_output_path, f)
        img = cv2.imread(p1, cv2.IMREAD_GRAYSCALE)
        res = img / 255 + np.random.normal(0, gaussian_sigma, img.shape)
        res = 255 * np.clip(res, 0, 1)

        cv2.imwrite(p2, res)    
if __name__ == "__main__":
    add_salt_noise()
    add_gaussian_noise()
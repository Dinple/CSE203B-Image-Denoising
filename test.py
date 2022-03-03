#test the performance of optimization alg defined in alg.py

from alg import denoise
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

source_path = "source"
salt_path = "salt"
gaussian_path = "gaussian"

# un-comment the following line for full test data
# file_names = os.listdir(source_path)

# for quick test using only 20 images
file_names = os.listdir(source_path)[:20]

img_num = len(file_names)

salt_dif_total = 0
gaussian_dif_total = 0

#compute the difference between original image and image with noises
for f in file_names:
    source = cv2.imread(os.path.join(source_path, f), cv2.IMREAD_GRAYSCALE)
    salt = cv2.imread(os.path.join(salt_path, f), cv2.IMREAD_GRAYSCALE)
    gaussian = cv2.imread(os.path.join(gaussian_path, f), cv2.IMREAD_GRAYSCALE)

    salt_dif_total += np.sum((source - salt) ** 2) ** 0.5
    gaussian_dif_total += np.sum((source - gaussian) ** 2) ** 0.5

#do the denoise and compute the difference
l_list = [0, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]

salt_dif_res_list = []
gaussian_dif_res_list = []

for l in l_list:
    print("for lambda: {}".format(l))
    salt_dif_res_total = 0
    gaussian_dif_res_total = 0

    for idx, f in enumerate(file_names):
        print("processing img({}/{})".format(idx+1, img_num), end="\r")
        source = cv2.imread(os.path.join(source_path, f), cv2.IMREAD_GRAYSCALE)
        salt = cv2.imread(os.path.join(salt_path, f), cv2.IMREAD_GRAYSCALE)
        gaussian = cv2.imread(os.path.join(gaussian_path, f), cv2.IMREAD_GRAYSCALE)

        salt_res = denoise(salt, l)
        gaussian_res = denoise(gaussian, l)
        salt_dif_res_total += np.sum((source - salt_res) ** 2) ** 0.5
        gaussian_dif_res_total += np.sum((source - gaussian_res) ** 2) ** 0.5
    print("")
    salt_dif_res_list.append(salt_dif_res_total / img_num)
    gaussian_dif_res_list.append(gaussian_dif_res_total / img_num)

#plot the difference between source and noised image (would be a horizontal line)
salt_dif_list = [salt_dif_total / img_num for i in range(len(l_list))]
gaussian_dif_list = [gaussian_dif_total / img_num for i in range(len(l_list))]

plt.plot(l_list, salt_dif_list, label="salt")
plt.plot(l_list, gaussian_dif_list, label='gaussian')
plt.plot(l_list, salt_dif_res_list, label="salt-opt")
plt.plot(l_list, gaussian_dif_res_list, label="gaussian-opt")
plt.legend()

plt.show()
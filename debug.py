from alg import denoise_nip
import cv2
import numpy as np
import os
import random

salt_prob = 0.02
source = cv2.imread(os.path.join("source", "00001.jpg"), cv2.IMREAD_GRAYSCALE)
mark = np.zeros(source.shape)
res = np.zeros(source.shape, np.uint8)
for i in range(res.shape[0]):
    for j in range(res.shape[1]):
        r = random.random()
        if r < salt_prob:
            res[i][j] = 0
            mark[i][j] = 1
        elif r > 1 - salt_prob:
            res[i][j] = 255
            mark[i][j] = 1
        else:
            res[i][j] = source[i][j]

thre = 64
i1, i2, i3, i4 = np.ones(res.shape), np.ones(res.shape), np.ones(res.shape), np.ones(res.shape)
img = res.astype(int)
i1[:, :-1] = np.abs(img[:, :-1] - img[:, 1:]) > thre
i2[:, 1:] = np.abs(img[:, 1:] - img[:, :-1]) > thre
i3[:-1, :] = np.abs(img[:-1, :] - img[1:, :]) > thre
i4[1:, :] = np.abs(img[1:, :] - img[:-1]) > thre
isolated = i1 * i2 * i3 * i4
imask = np.ones(img.shape) - np.ones(img.shape) * isolated

import ipdb; ipdb.set_trace()
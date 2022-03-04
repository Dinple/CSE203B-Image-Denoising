import cvxpy as cp 
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

#input: source image(hxw)
#output: denoised image(hxw)

def denoise(img, l=1):
    h, w = img.shape
    var = cp.Variable((h, w))
    
    #fidelity term
    obj1 = cp.sum_squares(var - img)

    #smooth term
    gradx = (var[:, 1:] - var[:, :-1]) / 2
    grady = (var[1:, :] - var[:-1, :]) / 2
    obj2 = cp.sum_squares(gradx) + cp.sum_squares(grady)

    #obj
    obj = obj1 + l * obj2

    #problem
    prob = cp.Problem(cp.Minimize(obj), [])
    prob.solve()
    res = var.value.astype(np.uint8)

    return res

if __name__ == "__main__":
    img_name = "00001.jpg"
    l = 5
    source = cv2.imread(os.path.join("source", img_name), cv2.IMREAD_GRAYSCALE)
    salt = cv2.imread(os.path.join("salt", img_name), cv2.IMREAD_GRAYSCALE)
    gaussian = cv2.imread(os.path.join("gaussian", img_name), cv2.IMREAD_GRAYSCALE)

    salt_res = denoise(salt, l)
    gaussian_res = denoise(gaussian, l)

    salt_dif = np.sum((source - salt) ** 2) ** 0.5
    salt_dif_res = np.sum((source - salt_res) ** 2) ** 0.5
    
    gaussian_dif = np.sum((source - gaussian) ** 2) ** 0.5
    gaussian_dif_res = np.sum((source - gaussian_res) ** 2) ** 0.5

    print("salt_dif: {:.2f}, salt_dif_res: {:.2f}".format(salt_dif, salt_dif_res))
    print("gaussian_dif: {:.2f}, gaussian_dif_res: {:.2f}".format(gaussian_dif, gaussian_dif_res))

    plt.subplot(311)
    plt.imshow(source, cmap ='gray')
    plt.subplot(312)
    plt.imshow(salt_res, cmap ='gray')
    plt.subplot(313)
    plt.imshow(salt, cmap ='gray')
    plt.show()

    plt.subplot(311)
    plt.imshow(source, cmap ='gray')
    plt.subplot(312)
    plt.imshow(gaussian_res, cmap ='gray')
    plt.subplot(313)
    plt.imshow(gaussian, cmap ='gray')

    plt.show()

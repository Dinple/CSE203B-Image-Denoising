import cvxpy as cp 
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

#input: 
#   img: source image(hxw)
#   l: lambda for the second term
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

#input: 
#   img: source image(hxw)
#   l: lambda for the second term
#   thre: the threshold to determine whether a pixel is isolated
#output: denoised image(hxw)

def denoise_nipv1(img, l=1, thre=64):

    i1, i2, i3, i4 = np.ones(img.shape), np.ones(img.shape), np.ones(img.shape), np.ones(img.shape)
    img = img.astype(int)
    i1[:, :-1] = np.abs(img[:, :-1] - img[:, 1:]) > thre
    i2[:, 1:] = np.abs(img[:, 1:] - img[:, :-1]) > thre
    i3[:-1, :] = np.abs(img[:-1, :] - img[1:, :]) > thre
    i4[1:, :] = np.abs(img[1:, :] - img[:-1]) > thre
    isolated = i1 * i2 * i3 * i4
    imask = np.ones(img.shape) - np.ones(img.shape) * isolated
    print("{:.2f}% pixel mask out".format(isolated.sum() / (imask.shape[0] * imask.shape[1])))

    h, w = img.shape
    var = cp.Variable((h, w))
    #fidelity term
    obj1 = cp.sum_squares(cp.multiply(imask, var - img))

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

def denoise_nipv2(img, l=1, thre=32):

    i1, i2, i3, i4 = np.ones(img.shape), np.ones(img.shape), np.ones(img.shape), np.ones(img.shape)
    img = img.astype(int)
    i1[:, :-1] = (np.abs(img[:, :-1] - img[:, 1:]) > thre).astype(int)
    i2[:, 1:] = (np.abs(img[:, 1:] - img[:, :-1]) > thre).astype(int)
    i3[:-1, :] = (np.abs(img[:-1, :] - img[1:, :]) > thre).astype(int)
    i4[1:, :] = (np.abs(img[1:, :] - img[:-1]) > thre).astype(int)
    isolated = (i1 + i2 + i3 + i4) >= 3

    print("{:.2f}% pixel mask out".format(isolated.sum() / (isolated.shape[0] * isolated.shape[1])))

    h, w = img.shape
    res = np.zeros((h, w))

    for i in range(h):
        for j in range(w):
            if isolated[i][j] == False:
                res[i][j] = img[i][j]
            else:
                neighboor = []
                if i > 0:
                    neighboor.append(img[i-1][j])
                if i < h - 1:
                    neighboor.append(img[i+1][j])
                if j > 0:
                    neighboor.append(img[i][j-1])
                if j < w - 1:
                    neighboor.append(img[i][j+1])
                
                if len(neighboor) == 2:
                    res[i][j] = (neighboor[0] + neighboor[1]) / 2
                elif len(neighboor) == 3:
                    res[i][j] = neighboor[1]
                elif len(neighboor) == 4:
                    neighboor.sort()
                    res[i][j] = (neighboor[1] + neighboor[2]) / 2
                else: 
                    import ipdb; ipdb.set_trace()
                    
    res = res.astype(np.uint8)

    return res

if __name__ == "__main__":
    img_name = "00001.jpg"
    l = 10
    source = cv2.imread(os.path.join("source", img_name), cv2.IMREAD_GRAYSCALE)
    salt = cv2.imread(os.path.join("salt", img_name), cv2.IMREAD_GRAYSCALE)
    gaussian = cv2.imread(os.path.join("gaussian", img_name), cv2.IMREAD_GRAYSCALE)

    salt_res = denoise_nipv2(salt, l)
    gaussian_res = denoise_nipv2(gaussian, l)

    # salt_dif = np.sum((source - salt) ** 2) ** 0.5
    # salt_dif_res = np.sum((source - salt_res) ** 2) ** 0.5
    
    # gaussian_dif = np.sum((source - gaussian) ** 2) ** 0.5
    # gaussian_dif_res = np.sum((source - gaussian_res) ** 2) ** 0.5

    salt_dif = np.abs(source - salt).sum()
    salt_dif_res = np.abs(source - salt_res).sum()
    
    gaussian_dif = np.abs(source - gaussian).sum()
    gaussian_dif_res = np.abs(source - gaussian_res).sum()

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

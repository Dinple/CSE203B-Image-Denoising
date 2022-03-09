import cvxpy as cp 
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

#input: source image(hxw)
#output: denoised image(hxw)
def edgeDetect(img, thre1=200, thre2=400):
    edges = cv2.Canny(img, thre1, thre2)
    plt.subplot(121),plt.imshow(img,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()
    return edges

#input: source image(hxw)
#output: denoised image(hxw)
def denoise_preserve_edge(img, l=1, thre1=200, thre2=400):
    h, w = img.shape
    var = cp.Variable((h, w))
    
    #fidelity term
    obj1 = cp.sum_squares(var - img)

    #smooth term
    gradx = (var[:, 1:] - var[:, :-1]) / 2
    grady = (var[1:, :] - var[:-1, :]) / 2
    obj2 = cp.sum_squares(gradx) + cp.sum_squares(grady)

    #constraints edge
    edgesDetect = edgeDetect(img, thre1, thre2) / 255
    edgesVal = cp.multiply(img, edgesDetect)
    edgesVar = cp.multiply(var, edgesDetect)
    constraints = [edgesVar == edgesVal]

    #obj
    obj = obj1 + l * obj2

    #problem
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve()
    res = var.value.astype(np.uint8)

    return res


#input: source image(hxw)
#output: denoised image(hxw)
def denoise_preserve_edgeGrad(img, l=1, thre1=200, thre2=400):
    h, w = img.shape
    var = cp.Variable((h, w))
    
    #fidelity term
    obj1 = cp.sum_squares(var - img)

    #smooth term
    gradx = (var[:, 1:] - var[:, :-1]) / 2
    grady = (var[1:, :] - var[:-1, :]) / 2
    obj2 = cp.sum_squares(gradx) + cp.sum_squares(grady)

    #constraints: edge gradient
    edgesDetect = edgeDetect(img, thre1, thre2) / 255
    imgGradx = (img[:, 1:] - img[:, :-1]) / 2
    imgGrady = (var[1:, :] - var[:-1, :]) / 2
    
    edgesGradxVal = cp.multiply(imgGradx, edgesDetect[:, :-1])
    edgesGradyVal = cp.multiply(imgGrady, edgesDetect[:-1, :])
    edgesGradxVar = cp.multiply(gradx, edgesDetect[:, :-1])
    edgesGradyVar = cp.multiply(grady, edgesDetect[:-1, :])

    constraints = [edgesGradxVar == edgesGradxVal, edgesGradyVar == edgesGradyVal]

    #obj
    obj = obj1 + l * obj2

    #problem
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve()
    res = var.value.astype(np.uint8)

    return res

if __name__ == "__main__":
    img_name = "00001.jpg"
    l = 5
    source = cv2.imread(os.path.join("source", img_name), cv2.IMREAD_GRAYSCALE)
    salt = cv2.imread(os.path.join("salt", img_name), cv2.IMREAD_GRAYSCALE)
    gaussian = cv2.imread(os.path.join("gaussian", img_name), cv2.IMREAD_GRAYSCALE)

    salt_res = denoise_preserve_edgeGrad(salt, l)
    gaussian_res = denoise_preserve_edgeGrad(gaussian, l)

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
    # plt.savefig('experiment-result/salt-edgeGradient.png')

    plt.subplot(311)
    plt.imshow(source, cmap ='gray')
    plt.subplot(312)
    plt.imshow(gaussian_res, cmap ='gray')
    plt.subplot(313)
    plt.imshow(gaussian, cmap ='gray')
    plt.show()
    # plt.savefig('experiment-result/gaussian-edgeGradient.png')

import os
import cv2
import numpy as np

source_path = "source"

#rename the images to: 00000.jpg, 00001.jpg ...
def rename():
    
    file_names = os.listdir(source_path)
    print(len(file_names))
    for idx, f in enumerate(file_names):
        p1 = os.path.join(source_path, f)
        p2 = os.path.join(source_path, "{:05d}.jpg".format(idx))
        os.rename(p1, p2)

#convert from rgb to grey image
def rgb2grey():
    file_names = os.listdir(source_path)
    for idx, f in enumerate(file_names):
        p = os.path.join(source_path, f)
        img = cv2.imread(p)
        img = img.mean(axis=-1).astype(np.uint8)
        cv2.imwrite(p, img)

if __name__ == "__main__":
    rename()
    rgb2grey()

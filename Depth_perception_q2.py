import cv2
import numpy as np
from matplotlib import pyplot as plt


def find_closest_index(val, arr):
    mindist = np.sqrt(sum((val - arr[0])**2))
    index = 0
    for i in range(1, len(arr)):
        dist = np.sqrt(sum((val - arr[i])**2))
        if dist < mindist:
            mindist = dist
            index = i
            if mindist == 0:
                return index
    return index


def find_disparity(img1, img2):
    dispMap = np.ndarray([img2.shape[0], img2.shape[1]-40])
    print(dispMap.shape)
    for i in range(dispMap.shape[0]):
        for j in range(dispMap.shape[1]):
            dispMap[i, j] = find_closest_index(img2[i, j], img1[i, j:j+40])
    return dispMap


if __name__ == "__main__":
    left_img = cv2.imread("im0.ppm")
    right_img = cv2.imread("im8.ppm")
    dMap = find_disparity(left_img, right_img)
    np.save("D_matrix", dMap)
    x = dMap.flatten()
    plt.style.use("ggplot")
    plt.hist(x, bins=40)
    plt.title("Disparity map histogram")
    plt.xlabel("Index distances")
    plt.ylabel("Number of pixels")
    plt.show()

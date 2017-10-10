import cv2
import random
import numpy as np
from matplotlib import pyplot as plt


def sample_img(img1, blocks):
    newimg = np.ndarray([8, img1.shape[1]*blocks])
    for i in range(blocks):
        x = random.randint(0, img1.shape[0] - 9)
        newimg[:, 768*i:768*(i+1)] = img1[x:x+8, :]
    return newimg


def power_iteration(A):
    b_k = np.random.rand(A.shape[0])
    for _ in range(2500):
        b_k1 = np.dot(A, b_k)
        b_k1_norm = np.linalg.norm(b_k1)
        b_k = b_k1 / b_k1_norm
    v1 = np.divide(np.dot(np.dot(A,b_k), b_k), np.dot(b_k, b_k))
    return b_k, v1


if __name__ == "__main__":
    random.seed(10)
    img = cv2.imread("IMG_1878.JPG")
    X_b = img[:, :, 0]
    X_g = img[:, :, 1]
    X_r = img[:, :, 2]
    # Modify this to change number of blocks per channel:
    block_num = 30
    new1 = sample_img(X_b, block_num)
    new2 = sample_img(X_g, block_num)
    new3 = sample_img(X_r, block_num)
    newimg = np.concatenate([new1, new2, new3], axis=1)
    newimg = newimg - np.mean(newimg, axis=1).reshape([8,1])
    cov_matrix = np.cov(newimg)
    y = cov_matrix.shape[0]
    Wt = np.ndarray([8,8])
    for i in range(y):
        A, v1 = power_iteration(cov_matrix)
        Wt[:, i] = A
        A = np.reshape(A, [8, 1])
        cov_matrix = cov_matrix - np.multiply(v1, np.dot(A, A.T))
    print(newimg.shape)
    plt.style.use("ggplot")
    plt.imshow(Wt.T, cmap="hot", interpolation="nearest")
    plt.show()

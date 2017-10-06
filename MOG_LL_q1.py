import cv2
import numpy as np
from random import randint
from matplotlib import pyplot as plt
import math


def patchaverage(img1, number):
    avgpatch = np.zeros([100, 100], dtype=float)
    for i in range(number):
        x = randint(0, 3555)
        y = randint(0, 6396)
        avgpatch += img1[x:x+100, y:y+100]/number
    plt.style.use("ggplot")
    plt.hist(avgpatch.ravel(), bins=75)
    plt.xlabel("Pixel intensities")
    plt.ylabel("Counts")
    plt.show()
    mean = avgpatch.mean()
    std = avgpatch.std()
    return avgpatch, mean, std


def loglikelihood(patch, mu, sd2):
    n = 10000
    patchmean = patch - mu
    patchmean = patchmean**2
    ll = -(n/2)*(np.log(2*math.pi) + np.log(sd2)) - (1/(2*sd2))*np.sum(patchmean)
    return ll


if __name__ == "__main__":
    img = cv2.imread("luddy.jpg")
    # Taking only the red channel:
    redimg = img[:, :, 2].astype(float)
    patch2, patch2mean, patch2std = patchaverage(redimg, 2)
    patch100, patch100mean, patch100std = patchaverage(redimg, 100)
    patch1000, patch1000mean, patch1000std = patchaverage(redimg, 1000)
    print("Patch 2: Mean: {0}, Std: {1}, Variance : {2}".format(patch2mean, patch2std, patch2std**2))
    print("Patch 100: Mean: {0}, Std: {1}, Variance : {2}".format(patch100mean, patch100std, patch100std**2))
    print("Patch 1000: Mean: {0}, Std: {1}, Variance : {2}".format(patch1000mean, patch1000std, patch1000std**2))
    # print(redimg.shape)
    # print(type(redimg))
    # cv2.imshow("luddy hall", redimg)
    # cv2.waitKey(0)
    print("Log likelihood for 2 patches: {0}". format(loglikelihood(patch2, patch2mean, patch2std**2)))
    print("Log likelihood for 100 patches: {0}".format(loglikelihood(patch100, patch100mean, patch100std**2)))
    print("Log likelihood for 1000 patches: {0}".format(loglikelihood(patch1000, patch1000mean, patch1000std**2)))

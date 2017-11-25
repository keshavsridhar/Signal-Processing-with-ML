import numpy as np
import random
import math
import cv2
from matplotlib import pyplot as plt
from scipy.stats import norm


def calculate_likelihood(x, mu, sigma2):
    return (1/(math.sqrt(2*math.pi*sigma2)))*math.exp((-0.5/sigma2)*(x - mu)**2)


class Gaussian:
    def __init__(self, data, p):
        self.mean = random.random()
        self.var = 1.0
        self.X = data
        self.likelihood = np.ndarray(data.shape)
        self.prior = p
        self.norm = 0
        self.N = data.shape[0]
        self.log_likelihood_updated = 0

    def calc_likelihood(self):
        temp = (-0.5/self.var) * (self.X - self.mean) ** 2
        num = list(map(math.exp, temp))
        denom = math.sqrt(2*math.pi*self.var)
        self.likelihood = np.divide(num, denom)

    def update_param(self):
        ui = np.divide((self.prior * self.likelihood), self.norm)
        ui = ui.reshape([ui.shape[0], 1])
        sum_norm = sum(ui)
        self.var = np.asscalar(sum(ui * (self.X - self.mean)**2)/sum_norm)
        self.mean = np.asscalar((np.dot(ui.T, self.X))/sum_norm)
        self.prior = np.asscalar(sum_norm/self.N)


def GMM(dMap, k):
    prior = 1 / k
    loglikes = []
    Gaussians = [Gaussian(dMap, prior) for x in range(k)]
    assert 1 - sum([g.prior for g in Gaussians]) <= 0.001
    old_ll = 0
    updated_ll = 1
    while old_ll - updated_ll != 0 :
        old_ll = updated_ll
        [g.calc_likelihood() for g in Gaussians]
        norm = sum([g.prior * g.likelihood for g in Gaussians])
        for g in Gaussians:
            g.norm = norm
            g.update_param()
            g.log_likelihood_updated = g.prior * g.likelihood

        # Calculating the log likelihood to check convergence:
        log_like = sum([g.log_likelihood_updated for g in Gaussians])
        updated_ll = round(sum(np.log(log_like)), 0)
        loglikes.append(updated_ll)
        print("Log likelihood : ", updated_ll)
    [print("final Mean: {0}, final SD: {1}, final Prior: {2}".format(g.mean, math.sqrt(g.var), g.prior)) for g in Gaussians]
    return Gaussians, loglikes


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


def smoothing_gibbs(dMap1, means, sds):
    data_dim = dMap1.shape
    smoothened_dMap1 = dMap1
    k = len(means)
    data_mat = dMap1.reshape(data_dim[0], data_dim[1], 1)
    dim = data_mat.shape
    i = 0
    while i < 100:
        i += 1
        _pr = norm.pdf(data_mat, means, sds)
        # Initialization of posterior as prior first:
        post = _pr
        # exact right:
        oPost = np.ones((dim[0], dim[1], k))
        oPost[:, 1:, :] = _pr[:, :-1, :]
        post = np.multiply(post, oPost)
        # exact left:
        oPost = np.ones((dim[0], dim[1], k))
        oPost[:, :-1, :] = _pr[:, 1:, :]
        post = np.multiply(post, oPost)
        # exact top:
        oPost = np.ones((dim[0], dim[1], k))
        oPost[:-1, :, :] = _pr[1:, :, :]
        post = np.multiply(post, oPost)
        # exact bottom:
        oPost = np.ones((dim[0], dim[1], k))
        oPost[1:, :, :] = _pr[:-1, :, :]
        post = np.multiply(post, oPost)
        # top left:
        oPost = np.ones((dim[0], dim[1], k))
        oPost[:-1, :-1, :] = _pr[1:, 1:, :]
        post = np.multiply(post, oPost)
        # top right:
        oPost = np.ones((dim[0], dim[1], k))
        oPost[:-1, 1:, :] = _pr[1:, :-1, :]
        post = np.multiply(post, oPost)
        # bottom left:
        oPost = np.ones((dim[0], dim[1], k))
        oPost[1:, :-1, :] = _pr[:-1, 1:, :]
        post = np.multiply(post, oPost)
        # bottom right:
        oPost = np.ones((dim[0], dim[1], k))
        oPost[1:, 1:, :] = _pr[:-1, :-1, :]
        post = np.multiply(post, oPost)
        # normalizing posterior using kxk ones matrix
        ONorm = np.ones((k, k))
        sumNorm = post.dot(ONorm)
        post = np.divide(post, sumNorm)
        smoothened_dMap1 = np.argmax(post, axis=2)
        _pr = post
    return smoothened_dMap1


if __name__ == "__main__":
    # Setting a seed:
    random.seed(10)
    left_img = cv2.imread("im0.ppm")
    right_img = cv2.imread("im8.ppm")
    # Uncomment this to find dMap and to save it (for faster loads later)
    # dMap = find_disparity(left_img, right_img)
    # np.save("D_matrix", dMap)
    dMap = np.load("D_matrix.npy")
    dMap1 = dMap.flatten()
    dMap1 = dMap1.reshape([dMap1.shape[0], 1])
    # Change k-value to pass to function as second argument:
    Mixtures, loglikes = GMM(dMap1, k=2)
    for i in range(dMap.shape[0]):
        for j in range(dMap.shape[1]):
            x = np.argmax([calculate_likelihood(dMap[i, j], m.mean, m.var) for m in Mixtures])
            # Disparity map with updated means:
            dMap[i, j] = int(Mixtures[x].mean)
    plt.style.use("ggplot")
    plt.imshow(dMap, cmap="gray")
    plt.show()
    g_means = [m.mean for m in Mixtures]
    g_sds = [math.sqrt(m.var) for m in Mixtures]
    smoothened_dMap = smoothing_gibbs(dMap, g_means, g_sds)
    plt.imshow(smoothened_dMap, cmap="gray")
    plt.show()

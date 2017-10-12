import numpy as np
import random
import math
from matplotlib import pyplot as plt


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
        updated_ll = round(sum(np.log(log_like)), 2)
        loglikes.append(updated_ll)
        print(updated_ll)
    [print("final Mean: {0}, final SD: {1}, final Prior: {2}".format(g.mean, math.sqrt(g.var), g.prior)) for g in Gaussians]
    return Gaussians, loglikes

if __name__ == "__main__":
    # Setting a seed:
    random.seed(10)
    dMap = np.load("D_matrix.npy")
    dMap1 = dMap.flatten()
    dMap1 = dMap1.reshape([dMap1.shape[0], 1])

    # Change k-value to pass to function as second argument:
    Mixtures, loglikes = GMM(dMap1, 2)
    for i in range(dMap.shape[0]):
        for j in range(dMap.shape[1]):
            x = np.argmax([calculate_likelihood(dMap[i, j], m.mean, m.var) for m in Mixtures])
            dMap[i, j] = int(Mixtures[x].mean)
    plt.style.use("ggplot")
    plt.imshow(dMap, cmap="gray")
    plt.show()

import numpy as np
import random
from matplotlib import pyplot as plt


class Cluster(object):
    def __init__(self):
        self.centroid = 0
        self.group = []
        self.indexes = []
        self.ss = 0
        self.prev_centroid = 0

    def add_to_cluster(self, x, i):
        self.group.append(x)
        self.indexes.append(i)

    def flush(self):
        self.group = []
        self.indexes = []

    def calc_mean(self):
        # To handle empty clusters adding a single element(previous centroid) to the cluster:
        self.centroid = (np.sum(self.group)+self.prev_centroid)/(len(self.group)+1)

    def calc_ss(self):
        self.ss = sum((np.array(self.group) - self.centroid)**2)


def kmeans(data, k):
    print(data.shape)
    k_dict = {}
    tss1 = []
    iterations = 0
    for z in k:
        Clusters = [Cluster() for i in range(z)]
        for i in Clusters:
            i.centroid = data[random.randint(0, 380), random.randint(0, 389)]
        prev_means = np.zeros([1, z])
        new_means = np.ones([1, z])
        while np.sum(prev_means - new_means) != 0:
            [c.flush() for c in Clusters]
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    x = np.argmin([(data[i, j] - l.centroid)**2 for l in Clusters])
                    Clusters[x].add_to_cluster(data[i, j], [i, j])
            prev_means = np.array([c.centroid for c in Clusters])
            [c.calc_mean() for c in Clusters]
            for r in Clusters:
                r.prev_centroid = r.centroid
            new_means = np.array([c.centroid for c in Clusters])
            [c.calc_ss() for c in Clusters]
            iterations += 1
        tss = sum([c.ss for c in Clusters])
        tss1.append(tss)
        print("# of Clusters: {0}, # of pixels in clusters: {1} ".format(z, [len(c.group) for c in Clusters]))
        print("Total sum of squares:", tss)
        k_dict[z] = Clusters
    return k_dict, tss1

if __name__ == "__main__":
    # Setting a seed:
    random.seed(10)
    # Loading saved disparity map from q2 (Execute q2.py first)
    dMap = np.load("D_matrix.npy")
    ks = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    Cluster_dict, tss = kmeans(dMap, ks)

    # Plotting depth map with different k-value:
    for k in Cluster_dict.values():
        for c in k:
            for i in c.indexes:
                dMap[i[0], i[1]] = int(c.centroid)
        plt.style.use("ggplot")
        plt.imshow(dMap, cmap="gray")
        plt.show()

    # Plotting the elbow graph to help choose k-value:
    plt.plot(ks, tss)
    plt.title("Finding best k -- Elbow method")
    plt.xlabel("k-value")
    plt.ylabel("Total sum of squares")
    plt.show()

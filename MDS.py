import scipy.io
import numpy as np
from matplotlib import pyplot as plt


def power_iteration(A):
    b_k = np.random.rand(A.shape[0])
    for _ in range(2500):
        b_k1 = np.dot(A, b_k)
        b_k1_norm = np.linalg.norm(b_k1)
        b_k = b_k1 / b_k1_norm
    v1 = np.divide(np.dot(np.dot(A,b_k), b_k), np.dot(b_k, b_k))
    return b_k, v1


def eigen_data(data_ker):
    Wt = np.ndarray([data_ker.shape[0], 2])
    eigvals = np.zeros(2)
    for i in range(2):
        A, v1 = power_iteration(data_ker)
        eigvals[i] = v1
        Wt[:, i] = A
        A = np.reshape(A, [data_ker.shape[0], 1])
        data_ker = data_ker - np.multiply(v1, np.dot(A, A.T))
    return Wt

if __name__ == "__main__":
    inp = scipy.io.loadmat("MDS_pdist.mat")
    L = inp['L']
    centering_matrix = np.identity(L.shape[0]) - (1/L.shape[0]) * np.ones([L.shape[0], L.shape[1]])
    W = (-1/2)*np.dot(centering_matrix, np.dot(L, centering_matrix))
    eig_vec = eigen_data(W)
    print(eig_vec)
    print(eig_vec.shape)
    plt.style.use("ggplot")
    plt.scatter(eig_vec[:, 0], eig_vec[:, 1])
    plt.show()

import scipy.io
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


def power_iteration(A):
    b_k = np.random.rand(A.shape[0])
    # b_k = np.ones(A.shape[0])

    for _ in range(25):
        # calculate the matrix-by-vector product Ab
        b_k1 = np.dot(A, b_k)
        # calculate the norm
        b_k1_norm = np.linalg.norm(b_k1)
        # re normalize the vector
        b_k = b_k1 / b_k1_norm
    v1 = np.divide(np.dot(np.dot(A,b_k), b_k), np.dot(b_k, b_k))
    return b_k, v1

if __name__ == "__main__":
    inputf = scipy.io.loadmat('flute.mat')
    matrix = inputf['X']
    # matrix = np.array([[2, -12], [1, -5]])
    # matrix = np.array([[25, 7], [7, 25]])
    # matrix = np.array([[9, -9], [-9, 9]])
    # print(np.linalg.norm([3, 4]))
    matrix = np.cov(matrix)
    b1, v1 = power_iteration(matrix)
    b1 = np.reshape(b1, [128, 1])
    # b1 = np.reshape(b1, [2, 1])
    B = matrix - np.multiply(v1, np.dot(b1, b1.T))
    b2, v2 = power_iteration(B)
    b2 = np.reshape(b2, [128, 1])
    eig_vectors = np.concatenate([b1, b2], axis=1)
    temporal_activation = np.dot(eig_vectors.T, inputf['X'])
    plt.style.use("ggplot")
    plt.imshow(inputf['X'])
    plt.show()
    plt.imshow(eig_vectors)
    plt.show()
    plt.imshow(temporal_activation)
    plt.show()
    # Projections:
    note1 = np.dot(inputf['X'].T,eig_vectors[:, 0]).reshape([1,143])
    note2 = np.dot(inputf['X'].T, eig_vectors[:, 1]).reshape([1, 143])
    plt.imshow(note1)
    plt.show()
    plt.imshow(note2)
    plt.show()
    notes = np.concatenate([note1, note2], axis=0)
    plt.imshow(notes)
    plt.show()
    # plt.plot(range(128),eig_vectors[:,0], range(128),eig_vectors[:,1])
    # plt.show()
    # print(eig_vectors[:,1].shape)

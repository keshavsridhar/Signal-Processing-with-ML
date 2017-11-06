from __future__ import division
import scipy.io
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def sigmoid(x):
    return 1.0/(1.0 + np.exp(np.negative(x)))


def power_iteration(A):
    b_k = np.random.rand(A.shape[0])
    for _ in range(2500):
        b_k1 = np.dot(A, b_k)
        b_k1_norm = np.linalg.norm(b_k1)
        b_k = b_k1 / b_k1_norm
    v1 = np.divide(np.dot(np.dot(A,b_k), b_k), np.dot(b_k, b_k))
    return b_k, v1


def eigen_data(data_ker):
    centered_data_ker = data_ker - np.mean(data_ker, axis=1).reshape([152, 1])
    Wt = np.ndarray([152, 3])
    eigvals = np.zeros(3)
    for i in range(3):
        A, v1 = power_iteration(centered_data_ker)
        eigvals[i] = v1
        Wt[:, i] = A
        A = np.reshape(A, [152, 1])
        centered_data_ker = centered_data_ker - np.multiply(v1, np.dot(A, A.T))
    # print(eigvals)
    return Wt


def calc_rbf(xi):
    return np.exp(-np.sum(xi, axis=0)).reshape([1, 152])


def pop_kernel(data):
    kernel_matrix = np.ndarray((data.shape[1], data.shape[1]))
    for i in range(kernel_matrix.shape[0]):
        kernel_matrix[i, :] = calc_rbf(np.power(data[:, i].reshape([2, 1]) - data[:, :], 2))
    return kernel_matrix


def sgd(y_hat, y):
    weights = np.random.randn(3)
    bias = np.random.randn()
    sse = []
    eta = 0.45
    pred = []
    for k in range(40000):
        sum_error = 0
        pred = []
        for i in range(y_hat.shape[0]):
            y_out = sigmoid(np.dot(weights.T, y_hat[i]) + bias)
            pred.append(y_out)
            sum_error += 0.5 * ((y_out - y[i])**2)
            delta = (y_out - y[i]) * y_out * (1 - y_out)
            weights -= eta * delta * y_hat[i]
            bias -= eta * delta
        print("Error: ", sum_error)
        sse.append(sum_error)
        if sum_error < 0.5:
            break
    plt.style.use("ggplot")
    plt.scatter(range(len(sse)), sse, marker=".")
    plt.xlabel("Iteration #")
    plt.ylabel("Sum of squared error")
    plt.show()
    return pred


def calc_acc(y_true, y_pred):
    total = len(y_true)
    temp_acc = sum(i == j for i, j in zip(y_true, y_pred))
    return temp_acc/total

if __name__ == "__main__":
    inp = scipy.io.loadmat("concentric.mat")
    # Setting a seed:
    np.random.seed(10)
    X = inp["X"]
    true_y = np.empty(152, dtype=int)
    true_y[:51] = 0
    true_y[51:] = 1
    # Generating the kernel NxN matrix:
    ker_matrix = pop_kernel(X)
    # Eigen decomposition on the kernel matrix:
    eigen_ker = eigen_data(ker_matrix)

    # Checking whether the alphas are normalized:
    # print(np.linalg.norm(eigen_ker[:, 0]))

    y = np.dot(eigen_ker.T, ker_matrix)
    y = y.T

    # Uncomment the following to check 3d plot to show that the points are linearly separable:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(y[:, 0], y[:, 1], y[:, 2])
    plt.show()

    y_pred = sgd(y, true_y)
    print(len(y_pred))
    y_pred = [0 if i < 0.5 else 1 for i in y_pred]
    print(y_pred)
    print(true_y)
    print("Accuracy: ", calc_acc(true_y, y_pred))

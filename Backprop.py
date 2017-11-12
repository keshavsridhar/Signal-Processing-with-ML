from __future__ import division
import scipy.io
from matplotlib import pyplot as plt
import numpy as np


def sigmoid(x):
    return 1.0/(1.0 + np.exp(np.negative(x)))


def sgd(data, y):
    weights_layer_1 = np.random.randn(6)
    bias_layer_1 = np.random.randn(3)
    output_layer_1 = np.empty(3)
    weights_layer_2 = np.random.randn(3)
    bias_layer_2 = np.random.randn()
    eta = 0.05
    sse = []
    pred = []
    for k in range(30000):
        sum_error = 0
        pred = []
        for i in range(data.shape[1]):
            # Forward pass:
            output_layer_1[0] = sigmoid(np.dot(weights_layer_1[:2], data[:, i]) + bias_layer_1[0])
            output_layer_1[1] = sigmoid(np.dot(weights_layer_1[2:4], data[:, i]) + bias_layer_1[1])
            output_layer_1[2] = sigmoid(np.dot(weights_layer_1[4:], data[:, i]) + bias_layer_1[2])
            y_out = sigmoid(np.dot(output_layer_1, weights_layer_2) + bias_layer_2)
            sum_error += 0.5 * ((y_out - y[i]) ** 2)
            pred.append(y_out)

            # Backward pass:
            delta_out = (y_out - y[i]) * y_out * (1 - y_out)
            weights_layer_1[0:2] -= eta*delta_out*weights_layer_2[0]*output_layer_1[0]*(1 - output_layer_1[0])*data[:, i]
            weights_layer_1[2:4] -= eta*delta_out*weights_layer_2[1]*output_layer_1[1]*(1 - output_layer_1[1])*data[:, i]
            weights_layer_1[4:6] -= eta*delta_out*weights_layer_2[2]*output_layer_1[2]*(1 - output_layer_1[2])*data[:, i]
            bias_layer_1[0] -= eta * delta_out * weights_layer_2[0] * output_layer_1[0] * (1 - output_layer_1[0])
            bias_layer_1[1] -= eta * delta_out * weights_layer_2[1] * output_layer_1[1] * (1 - output_layer_1[1])
            bias_layer_1[2] -= eta * delta_out * weights_layer_2[2] * output_layer_1[2] * (1 - output_layer_1[2])
            weights_layer_2 -= eta * delta_out * output_layer_1
            bias_layer_2 -= eta * delta_out
        print("Error: ", sum_error)
        sse.append(sum_error)
        if sum_error < 0.05:
            break
    return pred, weights_layer_2, bias_layer_2, weights_layer_1, bias_layer_1, sse


def calc_acc(y_true, y_pred):
    total = len(y_true)
    temp_acc = sum(i == j for i, j in zip(y_true, y_pred))
    return temp_acc/total


if __name__ == "__main__":
    inp = scipy.io.loadmat("concentric.mat")
    # Setting a seed:
    np.random.seed(10)
    X = inp["X"]
    print(X.shape)
    true_y = np.empty(152, dtype=int)
    true_y[:51] = 0
    true_y[51:] = 1
    y_pred, w2, b2, w1, b1, squared_error = sgd(X, true_y)
    y_pred = [0 if i < 0.5 else 1 for i in y_pred]
    print("Predicted: ", y_pred)
    print("Ground truth:", true_y)
    print("Accuracy: ", calc_acc(true_y, y_pred)*100, " percent")
    print("Weights 1st layer: ", w1, "Weights 2nd layer: ", w2)
    print("Bias 1st layer:", b1, "Bias 2nd layer: ", b2)
    plt.style.use("ggplot")
    plt.scatter(range(len(squared_error)), squared_error, marker=".")
    plt.xlabel("Iteration #")
    plt.ylabel("Sum of squared error")
    plt.show()

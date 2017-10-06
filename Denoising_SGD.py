from __future__ import division
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt


def sigmoid(matrix):
    return 1.0/(1.0 + np.exp(np.negative(matrix)))

if __name__ == "__main__":
    # Setting a random seed  to test results
    np.random.seed(10)

    source_img = cv2.imread("sg_train.jpg")
    noisy_img = cv2.imread("sgx_train.jpg")
    assert np.all([x == 0 for x in (2*source_img[:, :, 0] - source_img[:, :, 1] - source_img[:, :, 2])])
    source_img = source_img[:, :, 0]
    noisy_img = noisy_img[:, :, 0]
    source_img = source_img[7:193, 7:193]

    # Randomized filter from ~N(0,1)
    filter = np.random.randn(15, 15)
    filter = np.ravel(filter).reshape([1, 225])
    X = np.ndarray([225, 34596], dtype=np.float64)
    sourceT = np.ndarray([1, 34596], dtype=int)

    for i in range(len(noisy_img)-14):
        for j in range(len(noisy_img)-14):
            X_ij = noisy_img[i:i+15, j:j+15]
            X_ij = np.ravel(X_ij).reshape([225, ])
            X[:, 186*i + j] = X_ij
            sourceT[0, 186*i + j] = source_img[i, j]

    sourceT = sourceT/255
    epochs = 5000
    lrate = 0.001  # 0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.15, 0.45
    iteration = 0
    alpha = 0.9  # 0.01, 0.05, 0.1, 0.5, 0.9
    prev_gradient = np.zeros([225, 1])
    error_mat = np.ndarray([epochs, 1])
    for i in range(epochs):
        fTdotX = np.dot(filter, X)
        # fTdotX = fTdotX  # .astype(float)
        output_img = sigmoid(fTdotX)
        err1 = (sourceT - output_img)
        diff_err1 = output_img - np.multiply(output_img, output_img)
        # print(err1.shape, diff_err1.shape) # - 1 x 34596 - 1 x 34596
        # print(output_img.shape) # - 1 x 34596
        # print((np.dot(output_img, (1 - output_img).T)).shape) # 1 x 1
        # print(np.dot(X, err1.T).shape) # 225 x 1
        # print((output_img * (1. - output_img)).shape)
        gradient = (2.0 / 34596.0) * np.dot(X, (err1.T * diff_err1.T))
        # Momentum update:
        gradient += alpha * prev_gradient
        prev_gradient = gradient
        # print(prev_gradient)
        # print(gradient.shape) # - 255 x 1
        # print(filter.shape) # - 1 x 255
        # print(gradient.T.shape) # 1 x 255
        filter += lrate * gradient.T
        error = (1.0/34596.0) * np.dot(err1, err1.T)
        if iteration % 100 == 0:
            print("Error: ", error)
        error_mat[iteration, 0] = error
        iteration += 1

    # gradient = (2.0/34596.0) * (np.dot(output_img, (1. - output_img).T)) * np.dot(X, err1.T)
    # gradient = (2.0 / 34596.0) * np.dot(X,  (err1.T * (output_img * (1. - output_img)).T))
    # print(filter)
    # print("filter : ", filter.shape)
    # print('ftdotX', fTdotX)
    # print('filter', filter[0, :5])
    # print(sourceT.shape)
    # print("Source : ", source_img.shape)
    # print("Input X : ", X.shape)

    # print("fTdotX :", fTdotX.shape)
    # print("SourceT : ", sourceT.shape)
    fTdotX = np.dot(filter, X)
    # fTdotX = (fTdotX - fTdotX.mean()) / fTdotX.std()
    # print(fTdotX.shape) # - 1 x 34596
    out = np.ndarray([186, 186], dtype=int)
    for k in range(186):
        out[k, ] = fTdotX[0, k * 186: (k * 186) + 186] / 255
    print(out)
    # out = (out - out.mean())/out.std()
    # print(out)
    plt.imshow(out, cmap='gray')
    plt.show()
    plt.style.use("ggplot")
    # plt.imshow(source_img, cmap='gray')
    # plt.show()
    # cv2.imshow("source", source_img)
    # cv2.waitKey(0)
    # print(len(error_mat))
    error_mat = [float(x) for x in error_mat]
    print(error_mat)
    plt.plot(range(epochs), error_mat)
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.show()

    # Test image denoising:
    testimg = cv2.imread("sgx_test.jpg")
    testimg = testimg[:, :, 0]
    Y = np.ndarray([225, 34596], dtype=np.float64)
    for i in range(len(testimg)-14):
        for j in range(len(testimg)-14):
            Y_ij = testimg[i:i+15, j:j+15]
            Y_ij = np.ravel(Y_ij).reshape([225, ])
            Y[:, 186*i + j] = Y_ij
    testftdotX = np.dot(filter, Y)
    out_test = np.ndarray([186, 186], dtype=int)
    for k in range(186):
        out_test[k, ] = testftdotX[0, k * 186: (k * 186) + 186] / 255
    print(out_test)
    # out_test = (out_test - out_test.mean()) / out_test.std()
    # print(out_test)
    plt.imshow(out_test, cmap='gray')
    plt.show()

import numpy as np
import cv2
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
    X = np.ndarray([225, 34596], dtype=np.float64)
    sourceT = np.ndarray([1, 34596], dtype=int)
    for i in range(len(noisy_img)-14):
        for j in range(len(noisy_img)-14):
            X_ij = noisy_img[i:i+15, j:j+15]
            X_ij = np.ravel(X_ij).reshape([225, ])
            X[:, 186*i + j] = X_ij
            sourceT[0, 186*i + j] = source_img[i, j]
    sourceT = sourceT/255.0
    _filter = np.random.randn(15, 15)
    _filter = _filter.flatten()
    _filter = _filter.reshape((225, 1))
    _filter_prev = None
    prev_gradient = 0
    gradient = 0
    epochs = 10000
    lrate = 0.1
    error_mat = np.ndarray([epochs, 1])
    Gi0 = np.identity(225)
    for i in range(epochs):
        fTdotX = _filter.T.dot(X)
        err1 = (sourceT - sigmoid(fTdotX))
        if i > 0:
            prev_gradient = gradient
        gradient = float(2) / 34596.0 * X.dot((err1 * sigmoid(fTdotX)*(1.-sigmoid(fTdotX))).T)
        if i > 1:
            p = _filter - _filter_prev
            v = gradient - prev_gradient
            u = (p/np.linalg.norm(p.T.dot(v))) - Gi0.dot(v)/(np.linalg.norm(v.T.dot(Gi0).dot(v)))
            Gip1 = Gi0 + (p.dot(p.T)/(np.linalg.norm(p.T.dot(v)))) - ((Gi0.dot(v)).dot(v.T).dot(Gi0)/np.linalg.norm(v.T.dot(Gi0).dot(v))) + (np.linalg.norm(v.T.dot(Gi0).dot(v))) * (u.dot(u.T))
            # _max = Gip1.max()
            # _norm = np.linalg.norm(Gip1)
            Gip1 = Gip1 * Gip1.max() / max(np.linalg.norm(Gip1), Gip1.max())
        _filter_prev = _filter
        if i > 1:
            _filter = _filter + Gip1.dot(gradient)
        else:
            _filter = _filter + lrate * gradient
        if i > 1:
            Gi0 = Gip1
        error = (1.0 / 34596.0) * np.dot(err1, err1.T)
        if i % 100 == 0:
            print("Error: ", error)
        error_mat[i, 0] = error

    # print("fTdotX :", fTdotX.shape)
    # print("SourceT : ", sourceT.shape)
    fTdotX = np.dot(_filter.T, X)
    # fTdotX = (fTdotX - fTdotX.mean()) / fTdotX.std()
    # print(fTdotX.shape) # - 1 x 34596
    out = np.ndarray([186, 186], dtype=int)
    for k in range(186):
        out[k, ] = fTdotX[0, k * 186: (k * 186) + 186] / 255
    # print(out)
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
    testftdotX = np.dot(_filter.T, Y)
    out_test = np.ndarray([186, 186], dtype=int)
    for k in range(186):
        out_test[k, ] = testftdotX[0, k * 186: (k * 186) + 186] / 255
    # print(out_test)
    # out_test = (out_test - out_test.mean()) / out_test.std()
    # print(out_test)
    plt.imshow(out_test, cmap='gray')
    plt.show()

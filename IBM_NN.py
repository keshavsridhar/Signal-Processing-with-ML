import scipy.io.wavfile as io
import math
import numpy as np
from matplotlib import pyplot as plt


def vectorize_stft(inpx):
    N = 1024
    F = np.ndarray([N, N], dtype=complex)
    for i in range(N):
        for k in range(N):
            F[i, k] = math.cos(2 * math.pi * i * k / N) - 1j * math.sin(2 * math.pi * i * k / N)

    Hann_window = np.zeros(N)
    for i in range(Hann_window.shape[0]):
        Hann_window[i] = 0.5 * (1 - math.cos(2 * math.pi * i / (N - 1)))
    X = np.ndarray([N, int(np.ceil(len(inpx) / N) * 2)])
    x1 = np.zeros(N)
    c = 0
    for i in range(0, len(inpx), int(N / 2)):
        x1[:len(inpx[i:i + N])] = inpx[i:i + N]
        X[:, c] = np.multiply(x1, Hann_window)
        c += 1

    FX = np.dot(F, X)
    FX = abs(FX[:513, :])
    return FX


def sigmoid(x):
    return 1.0/(1.0 + np.exp(np.negative(x)))


def d_sigmoid(x):
    s = sigmoid(x)
    return s * (1-s)


def sgd(X, Mask_M):
    n = X.shape[1]
    d = X.shape[0]
    weights_layer_1 = np.random.normal(0, 0.1, (50, (d + 1)))
    weights_layer_2 = np.random.normal(0, 0.1, (d, (50 + 1)))
    reg_factor = 4
    eta = 0.01
    i = 0
    while i < 10000:
        # Forward pass:
        x0_layer1 = np.vstack((np.ones((1, n)), X))
        z2 = weights_layer_1.dot(x0_layer1)
        x0_layer2 = np.vstack((np.ones((1, n)), sigmoid(z2)))
        z3 = weights_layer_2.dot(x0_layer2)
        y_hat = sigmoid(z3)
        # Backward pass:
        error = y_hat - Mask_M
        d2 = np.dot(weights_layer_2[:, 1:].T, error) * d_sigmoid(z2)
        delta2 = np.dot(error, x0_layer2.T) + reg_factor * np.hstack((np.zeros((d, 1)), weights_layer_2[:, 1:]))
        delta1 = np.dot(d2, x0_layer1.T) + reg_factor * np.hstack((np.zeros((50, 1)), weights_layer_1[:, 1:]))
        weights_layer_1 = weights_layer_1 - eta * delta1 / n
        weights_layer_2 = weights_layer_2 - eta * delta2 / n
        acc = 1 - np.abs(np.round(y_hat) - Mask_M).mean()
        i += 1
        print('Iteration #: ', i, ' Accuracy: ', acc)
    return weights_layer_1, weights_layer_2


def calc_SNR(yhat, y):
    return 10*np.log10(y.var()/(y-yhat[:50791]).var())

if __name__ == "__main__":
    # Setting a seed:
    np.random.seed(10)

    # Reading training data:
    s = io.read("trs.wav")
    n = io.read("trn.wav")
    x = s[1] + n[1]
    print(len(x))
    X = vectorize_stft(x)
    S = vectorize_stft(s[1])
    N = vectorize_stft(n[1])
    print(X.shape)
    print(S.shape)
    print(N.shape)
    Mask_M = np.ndarray([S.shape[0], S.shape[1]])
    print(Mask_M.shape)
    for i in range(Mask_M.shape[0]):
        for j in range(Mask_M.shape[1]):
            if S[i, j] > N[i, j]:
                Mask_M[i, j] = 1
            else:
                Mask_M[i, j] = 0
    # print(Mask_M)
    # Uncomment to check mask:
    # plt.imshow(Mask_M)
    # plt.show()
    weights_layer_1, weights_layer_2 = sgd(X, Mask_M)

    # Reading the test notes:
    test_x = io.read("tex.wav")
    test_s = io.read("tes.wav")
    print(test_s)
    print(test_x)
    test_x_stft = vectorize_stft(test_x[1])
    # Doing a forward pass of the network with weights learned from training
    # to get the mask for test_x:
    n = test_x_stft.shape[1]
    x0_layer1 = np.vstack((np.ones((1, n)), test_x_stft))
    z2 = weights_layer_1.dot(x0_layer1)
    x0_layer2 = np.vstack((np.ones((1, n)), sigmoid(z2)))
    z3 = weights_layer_2.dot(x0_layer2)
    h = sigmoid(z3)
    Mask_test = np.round(h)
    # Getting the speech spectrogram:
    S_test = np.multiply(Mask_test, test_x_stft)
    print(S_test.shape)
    S_test = np.concatenate((S_test, -S_test.imag[:511]), axis=0)
    print(S_test.shape)

    # Inverse STFT:
    Finv = np.ndarray([1024, 1024], dtype=complex)
    for i in range(1024):
        for k in range(1024):
            Finv[i, k] = math.cos(2 * math.pi * i * k / 1024) + math.sin(2 * math.pi * i * k / 1024) * 1j
    X_hat = np.dot(Finv.T, S_test)
    X_hat = X_hat.real
    X_hat = np.transpose(X_hat)
    print(X_hat.shape)
    Y = np.zeros(test_x[1].shape[0]+512)
    Y[0:512] = X_hat[0, :512]
    for i in range(X_hat.shape[0] - 1):
        # print((X_hat[i, 512:] + X_hat[i + 1, :512]).shape)
        Y[512 * (i + 1):512 * (i + 2)] = X_hat[i, 512:] + X_hat[i + 1, :512]
    Y = (Y / np.max(Y)) * 32768
    Y = Y.astype(np.int16)
    print(Y)
    print(Y.shape)
    io.write("s_hat.wav", rate=test_x[0], data=Y)
    # Y = ((Y - Y.min()) / (np.max(Y) - np.min(Y)))
    # y_true = ((test_s[1] - test_s[1].min()) / (np.max(test_s[1]) - np.min(test_s[1])))
    Y = Y/Y.var()
    y_true = test_s[1]/test_s[1].max()
    print(calc_SNR(Y, y_true))

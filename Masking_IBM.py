import scipy.io.wavfile
import numpy as np
import math
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
    # print(X.shape)
    x1 = np.zeros(N)
    c = 0
    for i in range(0, len(inpx), int(N / 2)):
        x1[:len(inpx[i:i + N])] = inpx[i:i + N]
        X[:, c] = np.multiply(x1, Hann_window)
        c += 1

    FX = np.dot(F, X)
    FX = abs(FX[:513, :157])

    return FX


def calc_SNR(yhat, y):
    num = 0
    denom = 0
    for i in range(len(y)):
        num += y[i] ** 2
        denom += (y[i] - yhat[i]) ** 2
    x = num / denom
    return 10*np.log10(x)

if __name__ == "__main__":
    piano = scipy.io.wavfile.read("data\piano.wav")
    ocean = scipy.io.wavfile.read("data\ocean.wav")
    print(piano[1].shape)
    print(ocean[1].shape)
    S = vectorize_stft(piano[1])
    N = vectorize_stft(ocean[1])
    print("S:", S.shape)
    print("N:", N.shape)
    # plt.imshow(S.real, interpolation="nearest")
    # plt.show()
    # plt.imshow(N.real, interpolation="nearest")
    # plt.show()
    X = S + N
    # print(X)
    # plt.imshow(X.real, interpolation="nearest")
    # plt.show()
    B_mask = np.zeros([513,157])
    for i in range(513):
        for j in range(157):
            if S[i, j] > N[i, j]:
                B_mask[i, j] = 1
            else:
                B_mask[i, j] = 0
    print(B_mask)
    new_S = np.multiply(B_mask, X)
    # print(new_S)
    new_S = np.concatenate((new_S, -new_S.imag[:511]), axis=0)
    # new_S.concatenate(-new_S.imag)
    # print(new_S.shape)
    # F-inverse:
    Finv = np.ndarray([1024, 1024], dtype=complex)
    for i in range(1024):
        for k in range(1024):
            Finv[i, k] = math.cos(2 * math.pi * i * k / 1024) + math.sin(2 * math.pi * i * k / 1024) * 1j
    X_hat = np.dot(Finv.T, new_S)
    X_hat = X_hat.real
    X_hat = np.transpose(X_hat)
    # print(X_hat.shape)
    Y = np.zeros(80384)
    Y[0:512] = X_hat[0, :512]
    for i in range(X_hat.shape[0] - 1):
        Y[512 * (i + 1):512 * (i + 2)] = X_hat[i, 512:] + X_hat[i + 1, :512]
    # print(Y)
    Y = (Y / np.max(Y)) * 32768
    Y = Y.astype(np.int16)
    scipy.io.wavfile.write("Piano_out_IBM.wav", rate=piano[0], data=Y)
    Y = ((Y - Y.min()) / (np.max(Y) - np.min(Y)))
    # print(Y)
    y_true = ((piano[1] - piano[1].min()) / (np.max(piano[1]) - np.min(piano[1])))
    # y_true = piano[1]/np.max(piano[1])
    # y_true = y_true.astype(np.int16)
    print(calc_SNR(Y, y_true))

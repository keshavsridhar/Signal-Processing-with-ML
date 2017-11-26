from matplotlib import pyplot as plt
import scipy.io.wavfile as io
import math
import numpy as np


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


def calc_PLSI(X, k=20, W=None):
    feats = X.shape[0]
    points = X.shape[1]
    if W is None:
        W = np.random.randint(0, high=10, size=(feats, k))
        OW = np.ones((feats, feats))
        update_W = True
    else:
        update_W = False
    H = np.dot(W.T, X)
    k_dim = H.shape[0]
    OH = np.ones((k_dim, k_dim))
    i = 0
    eps = 0.0001
    while i < 2500:
        if update_W:
            W_del = np.divide(np.dot(X, H.T), np.dot(W, np.dot(H, H.T)) + eps)
            W = np.multiply(W, W_del)
            W = W / np.dot(OW, W)
            # print('updating w')
        H_del = np.divide(np.dot(W.T, X), np.dot(np.dot(W.T, W), H) + eps)
        H = np.multiply(H, H_del)
        H = H / np.dot(OH, H)
        temp = np.subtract(X, np.dot(W, H))
        err = np.linalg.norm(np.dot(temp.T, temp), ord='fro') / points
        print('Iteration #: ', i, 'Err: ', err)
        i += 1
    return W, H


def calc_SNR(yhat, y):
    return 10*np.log10(y.var()/(y-yhat[:y.shape[0]]).var())


if __name__ == "__main__":
    # Setting a seed:
    np.random.seed(10)
    # Reading training data:
    s = io.read("trs.wav")
    n = io.read("trn.wav")
    S = vectorize_stft(s[1])
    N = vectorize_stft(n[1])
    test_s = io.read("tes.wav")
    test_x = io.read("tex.wav")
    test_x_stft = vectorize_stft(test_x[1])
    test_s_stft = vectorize_stft(test_s[1])
    S = S/np.sqrt(S.var())
    N = N/np.sqrt(N.var())
    test_x_stft = test_x_stft/np.sqrt(test_x_stft)
    test_s_stft = test_s_stft/np.sqrt(test_s_stft)
    Ws, Hs = calc_PLSI(S)
    Wn, Hn = calc_PLSI(N)
    Wtot = np.hstack((Ws, Wn))
    _, Htot = calc_PLSI(np.absolute(test_x_stft), W=Wtot)
    Y = np.dot(Wtot, Htot)
    # 20 basis vectors
    Mask_M = np.divide(np.dot(Ws, Htot[:20]), Y)
    S_pred = np.multiply(Mask_M, test_x_stft)
    print(S_pred)
    print(S_pred.shape)
    S_pred = np.concatenate((S_pred, -S_pred.imag[:511]), axis=0)
    # Inverse STFT:
    Finv = np.ndarray([1024, 1024], dtype=complex)
    for i in range(1024):
        for k in range(1024):
            Finv[i, k] = math.cos(2 * math.pi * i * k / 1024) + math.sin(2 * math.pi * i * k / 1024) * 1j
    X_hat = np.dot(Finv.T, S_pred)
    X_hat = X_hat.real
    X_hat = np.transpose(X_hat)
    print(X_hat.shape)
    Y = np.zeros(test_x[1].shape[0] + 512)
    Y[0:512] = X_hat[0, :512]
    for i in range(X_hat.shape[0] - 1):
        Y[512 * (i + 1):512 * (i + 2)] = X_hat[i, 512:] + X_hat[i + 1, :512]
    Y = (Y / np.max(Y)) * 32768
    Y = Y.astype(np.int16)
    print(Y)
    print(Y.shape)
    io.write("s_hat_q3.wav", rate=test_x[0], data=Y)
    Y = Y / Y.var()
    y_true = test_s[1] / test_s[1].max()
    print(calc_SNR(Y, y_true))

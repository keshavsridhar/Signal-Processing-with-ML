import scipy.io.wavfile
import numpy as np


def power_iteration(A):
    b_k = np.random.rand(A.shape[0])
    for _ in range(2500):
        b_k1 = np.dot(A, b_k)
        b_k1_norm = np.linalg.norm(b_k1)
        b_k = b_k1 / b_k1_norm
    v1 = np.divide(np.dot(np.dot(A,b_k), b_k), np.dot(b_k, b_k))
    return b_k, v1


def tanh(data):
    return np.tanh(data)

if __name__ == "__main__":
    inp = scipy.io.wavfile.read(r"data\x_ica_1.wav")
    X = np.ndarray([20, inp[1].shape[0]])
    X[0] = inp[1]
    print(X[0])
    for i in range(2, 21):
        inp = scipy.io.wavfile.read(r"data\x_ica_"+str(i)+".wav")
        X[i-1] = inp[1]
    centered_X = X - np.mean(X, axis=1).reshape([20,1])
    cov_X = np.cov(centered_X)
    Wt = np.ndarray([20, 20])
    eigvals = np.zeros(20)
    for i in range(20):
        A, v1 = power_iteration(cov_X)
        eigvals[i] = v1
        Wt[:, i] = A
        A = np.reshape(A, [20, 1])
        cov_X = cov_X - np.multiply(v1, np.dot(A, A.T))
    # print(eigvals)
    # I chose the first 3 eigen vectors
    # First 3 eigen values dominate the rest, so most variance explained by the first 3:
    whitened_vectors = np.divide(Wt[:, :3], np.sqrt(eigvals[:3]))
    whitened_projection = np.dot(whitened_vectors.T, centered_X)

    # ICA:
    # Initialization:
    Z = whitened_projection
    I = np.identity(3)
    N = Z.shape[1]
    W = np.identity(3, dtype=float)
    Y = np.dot(W, Z)
    rho = 0.000001
    threshold = 1
    # Learning:
    while threshold != 0:
        old_weight = W
        delta_W = np.dot((N*I - np.dot(tanh(Y), np.power(Y, 3).T)), W)
        W = W + rho*delta_W
        Y = np.dot(W, Z)
        new_weight = W
        threshold = round(np.sum(old_weight - new_weight), 5)
        print(np.sum(old_weight - new_weight))
    for i in range(3):
        Y[i] = (Y[i] / np.max(Y[i])) * 32768
    Y = Y.astype(np.int16)
    scipy.io.wavfile.write("first.wav", rate=inp[0], data=Y[0, :])
    scipy.io.wavfile.write("second.wav", rate=inp[0], data=Y[1, :])
    scipy.io.wavfile.write("third.wav", rate=inp[0], data=Y[2, :])

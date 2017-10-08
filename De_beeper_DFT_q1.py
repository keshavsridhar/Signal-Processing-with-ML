import scipy.io.wavfile
import numpy as np
import math
from matplotlib import pyplot as plt

if __name__ == "__main__":
    inp = scipy.io.wavfile.read("x.wav")
    inpx = inp[1]
    N = 1600
    F = np.ndarray([N, N], dtype=complex)
    for i in range(N):
        for k in range(N):
            F[i, k] = math.cos(2*math.pi*i*k/N) - 1j*math.sin(2*math.pi*i*k/N)

    Hann_window = np.zeros(N)
    for i in range(Hann_window.shape[0]):
        Hann_window[i] = 0.5 * (1 - math.cos(2*math.pi*i/(N-1)))
    X = np.ndarray([N, int(np.ceil(len(inpx)/N)*2)])
    print(X.shape)
    x1 = np.zeros(N)
    c = 0
    for i in range(0, len(inpx), int(N/2)):
        x1[:len(inpx[i:i+N])] = inpx[i:i + N]
        X[:, c] = np.multiply(x1, Hann_window)
        c += 1

    FX = np.dot(F, X)
    plt.style.use("ggplot")
    plt.imshow(FX.real)
    plt.show()
    plt.imshow(FX.imag)
    plt.show()

    # Row 200 and Row 1400 - Setting to zero:
    FX[199] = np.zeros(int(math.ceil(len(inpx) / N) * 2))
    FX[200] = np.zeros(int(math.ceil(len(inpx)/N)*2))
    FX[201] = np.zeros(int(math.ceil(len(inpx) / N) * 2))
    FX[1399] = np.zeros(int(math.ceil(len(inpx) / N) * 2))
    FX[1400] = np.zeros(int(math.ceil(len(inpx)/N) * 2 ))
    FX[1401] = np.zeros(int(math.ceil(len(inpx) / N) * 2))

    Finv = np.ndarray([N, N], dtype=complex)
    for i in range(N):
        for k in range(N):
            Finv[i, k] = math.cos(2*math.pi*i*k/N) + math.sin(2*math.pi*i*k/N)*1j
    X_hat = np.dot(Finv.T, FX)
    X_hat = X_hat.real
    X_hat = np.transpose(X_hat)
    print(X_hat.shape)
    Y = np.zeros(64000)
    Y[0:800] = X_hat[0, :800]
    for i in range(X_hat.shape[0]-1):
        Y[800*(i+1):800*(i+2)] = X_hat[i,800:] + X_hat[i+1,:800]
    Y = (Y/np.max(Y)) * 32768
    Y = Y.astype(np.int16)
    scipy.io.wavfile.write("x_output.wav", inp[0], Y)

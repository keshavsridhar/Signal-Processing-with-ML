import scipy.io as matload
from matplotlib import pyplot as plt
import math
import numpy as np
import pandas as pd


def power_iteration(A):
    b_k = np.random.rand(A.shape[0])
    for _ in range(2500):
        b_k1 = np.dot(A, b_k)
        b_k1_norm = np.linalg.norm(b_k1)
        b_k = b_k1 / b_k1_norm
    v1 = np.divide(np.dot(np.dot(A,b_k), b_k), np.dot(b_k, b_k))
    return b_k, v1


def vectorize_stft(data):
    N = 64
    dft = np.ndarray([N, N], dtype=complex)
    for i in range(N):
        for k in range(N):
            dft[i, k] = math.cos(2*math.pi*i*k/N) - 1j*math.sin(2*math.pi*i*k/N)
    blackman_window = np.zeros(N)
    for i in range(len(blackman_window)):
        blackman_window[i] = 0.42 - 0.5*(math.cos(2*math.pi*i/(N-1))) + 0.08*math.cos(4*math.pi*i/(N-1))
    hopsize = 48
    sample = 768
    # print("window:", blackman_window.shape)
    # print("stft:", X.shape)
    # print("datapoint:", data[:, 0, 0].shape)
    X_data_stft = np.ndarray([255, data.shape[2]])
    for k in range(data.shape[2]):
        X_sample = np.zeros([255])
        for j in range(3):
            X = np.zeros([N, int(sample/hopsize)+1])
            c = 0
            temp_x = np.zeros(64)
            for i in range(0, sample, hopsize):
                if len(data[i:i+64, j, k]) < 64:
                    temp_x[:len(data[i:i+64, j, k])] = data[i:i+64, j, k]
                    X[:, c] = np.multiply(temp_x, blackman_window)
                else:
                    X[:, c] = np.multiply(data[i:i+64, j, k], blackman_window)
                c += 1
            X = np.dot(dft, X)
            X = abs(X[:33])
            # print(X.shape)
            # plt.style.use("ggplot")
            # plt.imshow(X.real, interpolation="nearest")
            # plt.show()
            X_flat = np.array(X[3:8])
            X_flat = X_flat.flatten()
            X_sample[85*j:85*j + 85] = X_flat
            X_data_stft[:, k] = X_sample
    return X_data_stft


def pca_data(data_pca, L, M, random_projection1):
    centered_data_pca = data_pca - np.mean(data_pca, axis=1).reshape([255, 1])
    train_cov_mat = np.cov(centered_data_pca)
    Wt = np.ndarray([255, M])
    eigvals = np.zeros(M)
    for i in range(M):
        A, v1 = power_iteration(train_cov_mat)
        eigvals[i] = v1
        Wt[:, i] = A
        A = np.reshape(A, [255, 1])
        train_cov_mat = train_cov_mat - np.multiply(v1, np.dot(A, A.T))
    projected_data = np.dot(Wt.T, centered_data_pca)

    # New projection on top of PCA:
    Y = np.sign(np.dot(random_projection1, projected_data))

    return Y


def calc_acc(y_true, y_pred):
    total = len(y_true)
    temp_acc = sum(i == j for i, j in zip(y_true, y_pred))
    return temp_acc/total


def knn(train_data, test_data, train_label, test_label, K):
    test_pred = []
    ks = []
    for i in range(test_data.shape[1]):
        hamm_dists = []
        for j in range(train_data.shape[1]):
            hamm_dist = sum(s1 != s2 for s1, s2 in zip(test_data[:, i], train_data[:, j]))
            hamm_dists.append(hamm_dist)
        for l in range(K):
            temp = np.argmin(hamm_dists)
            ks.append(train_label[temp])
            hamm_dists[temp] = test_data.shape[0] + 1
        keys, counts = np.unique(ks, return_counts=True)
        test_pred.append(np.argmax(counts)+1)
    test_pred = np.array(test_pred)
    test_label = test_label.reshape(28)
    acc = calc_acc(test_label, test_pred)
    return acc


if __name__ == "__main__":
    inp = matload.loadmat("data\eeg.mat")
    results_df = pd.DataFrame(columns=list('MLKA'))
    # Setting a seed to test results:
    np.random.seed(10)
    X_train = inp['x_train']
    y_train = inp['y_train']
    X_test = inp['x_te']
    y_test = inp['y_te']
    print("Train data shape:", X_train.shape)
    print("Test data shape:", X_test.shape)

    # Train data STFT:
    train_data_stft = vectorize_stft(X_train)
    # Test data STFT:
    test_data_stft = vectorize_stft(X_test)
    print("Train_data_stft shape:", train_data_stft.shape)

    # Set bounds for K, L, M:
    # Results were captured for the following values of K, L and M:
    # K = 15
    # L = 10
    # M = 20

    # This set of K, L and M is to show the working(runs faster):
    K = 5
    L = 5
    M = 5
    for m in range(2, M):
        for l in range(2, L):
            for k in range(2, K):
                random_projection = np.random.rand(l, m)
                # Making sure each row vector is normalized such that they are unit vectors:
                proj_sums = np.linalg.norm(random_projection, axis=1).reshape([l, 1])
                random_projection = np.divide(random_projection, proj_sums)
                # Train data PCA + Random projection:
                train_sign_data = pca_data(train_data_stft, l, m, random_projection)
                # print("Train data_random_projection shape:", train_sign_data.shape)
                # print("Train label shape:", y_train.shape)
                # Test data PCA + Random projection:
                test_sign_data = pca_data(test_data_stft, l, m, random_projection)
                # print("Test data_random_projection shape:", test_sign_data.shape)
                # print("Test label shape:", y_test.shape)
                accuracy = knn(train_sign_data, test_sign_data, y_train, y_test, k)
                # print(M, L, K, accuracy)
                results_df.loc[-1] = [m, l, k, accuracy]
                results_df.index += 1
    print(results_df)
    print(max(results_df['A']))
    # Write output to csv file:
    # results_df.to_csv("Q4_knn_results")

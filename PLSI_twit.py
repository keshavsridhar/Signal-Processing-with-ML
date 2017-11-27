import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat


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
    while i < 1500:
        i += 1
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

    return W, H


def sgd(Htr, YteMat, YtrMat):
    feats = Htr.shape[0]
    points = Htr.shape[1]
    ys = YtrMat.shape[0]
    w = np.random.normal(0, 1, (feats + 1, ys))
    X = np.vstack((np.ones((1, points)), Htr))
    y = YtrMat.T
    norm_ones = np.ones((ys, ys))
    true_y = np.argmax(y, axis=1)
    eta = 5
    i = 0
    while i < 15000:
        out_layer = np.dot(X.T, w)
        z2 = np.exp(out_layer)
        h_smax = z2 / np.dot(z2, norm_ones)
        y_pred = np.argmax(h_smax, axis=1)
        delta_w = np.dot(X, (y - h_smax)) / points
        w = w + eta * delta_w
        accuracy = (true_y == y_pred).astype(int).mean()
        print('Iteration #:', i, 'Accuracy: ', accuracy)
        i += 1
    # Feeding forward for test after learning weights:
    n_te = Hte.shape[1]
    X_te = np.vstack((np.ones((1, n_te)), Hte))
    y_te = YteMat.T
    true_y = np.argmax(y_te, axis = 1)
    out_layer = np.dot(X_te.T, w)
    z2 = np.exp(out_layer)
    h_smax = z2 / np.dot(z2, norm_ones)
    y_pred = np.argmax(h_smax, axis = 1)
    accuracy = np.array((true_y == y_pred)).astype(int).mean()
    print('Test accuracy: ', accuracy)

if __name__ == "__main__":
    # Setting a seed:
    np.random.seed(10)
    twitMat = loadmat("twitter.mat")
    # print(twitMat)
    Xtr = twitMat['Xtr']
    YtrMat = twitMat['YtrMat']
    Xte = twitMat['Xte']
    YteMat = twitMat['YteMat']
    topics = 50
    # Getting activation and basis from train data:
    Wtr, Htr = calc_PLSI(Xtr, topics)
    # Estimating for test data:
    Wte, Hte = calc_PLSI(Xte, topics, Wtr)
    # Perceptron training and testing:
    sgd(Htr, YteMat, YtrMat)

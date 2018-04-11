import numpy as np

def train_test_split(X, Y, train_split):

    n = X.shape[0]

    train_count = int(n * train_split)

    test_count = n - train_count

    ids = np.random.permutation(n)

    train_ids = ids[0:train_count]
    test_ids = ids[-test_count:]

    X_train = X[train_ids]
    X_test = X[test_ids]

    Y_train = Y[train_ids]
    Y_test = Y[test_ids]

    return X_train, X_test, Y_train, Y_test

def plot_contour(X1_train, X2_train, clf):
    pl.plot(X1_train[:,0], X1_train[:,1], "ro")
    pl.plot(X2_train[:,0], X2_train[:,1], "bo")
    pl.scatter(clf.sv[:,0], clf.sv[:,1], s=100, c="g")

    X1, X2 = np.meshgrid(np.linspace(-6,6,50), np.linspace(-6,6,50))
    X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
    Z = clf.project(X).reshape(X1.shape)
    pl.contour(X1, X2, Z, [0.0], colors='k', linewidths=1, origin='lower')
    pl.contour(X1, X2, Z + 1, [0.0], colors='grey', linewidths=1, origin='lower')
    pl.contour(X1, X2, Z - 1, [0.0], colors='grey', linewidths=1, origin='lower')

    pl.axis("tight")
    pl.show()
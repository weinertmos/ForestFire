import numpy as np


def import_data():
    """
    Import raw data from two numpy arrays X.npy and y.npy. 
    Set how train and test data are to be split for fix splits.
    Returns train/test splits as well as number of features.
    """
    X = np.load('X.npy')
    X_train = X[:500]
    X_test = X[500:800]
    y = np.load('y.npy')
    y_train = y[:500]
    y_test = y[500:800]
    n_feat = len(X[0])

    return X_test, X_train, y_test, y_train, n_feat

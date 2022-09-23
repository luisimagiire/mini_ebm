import numpy as np


def rmse(target, mean):
    diff = target - mean
    diff = np.power(diff, 2)
    return np.mean(np.sqrt(diff))


def make_bulk_prediction(X: np.ndarray, tree):
    """
    Helper function to make predictions for bulk array
    :param X:
    :param tree: Decision tree model
    :return:
    """
    pred = lambda x: tree.predict(tree, x)
    fmt = np.asarray(X) if X.ndim == 2 else np.asarray(X).reshape(-1, 1)
    return np.array([pred(fmt[i, :]) for i in range(X.shape[0])])

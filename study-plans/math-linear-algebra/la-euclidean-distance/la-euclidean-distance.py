import numpy as np

def euclidean_distance(x, y):
    """
    Returns: float, the Euclidean distance between x and y.
    """
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)

    return np.sqrt(np.sum((x-y)**2))
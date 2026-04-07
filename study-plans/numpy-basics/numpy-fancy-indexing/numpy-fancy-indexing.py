import numpy as np

def select_by_index(arr, indices, axis):
    """
    Returns: 2D ndarray of float64
    """
    arr = np.asarray(arr, np.float64)

    sub = np.take(arr, indices, axis=axis)

    return sub
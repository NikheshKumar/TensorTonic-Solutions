import numpy as np

def select_by_index(arr, indices, axis):
    """
    Returns: 2D ndarray of float64
    """
    arr = np.asarray(arr, dtype=np.float64)

    new_index = [slice(None)] *arr.ndim
    new_index[axis] = indices
    subarray = arr[tuple(new_index)]

    return subarray
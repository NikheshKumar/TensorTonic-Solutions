import numpy as np

def extract_subarray(arr, row_start, row_stop, col_start, col_stop):
    """
    Returns: 2D ndarray of float64
    """
    new = np.asarray(arr, dtype=np.float64)
    new = new[row_start:row_stop, col_start:col_stop]
    
    return new

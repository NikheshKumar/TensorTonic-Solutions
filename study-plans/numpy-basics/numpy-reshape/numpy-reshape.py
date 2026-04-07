import numpy as np

def reshape_array(data, operation):
    """
    Returns: ndarray of float64 with shape determined by the operation
    """
    data = np.asarray(data, dtype=np.float64)
    
    if operation=="flatten":
        return data.ravel()
        
    if operation=="transpose":
        return data.T

    if operation=="add_batch":
        return data[np.newaxis, ...]
        
import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    """
    Normalize a 2D matrix along specified axis using specified norm.
    """
    # Write code here
    matrix = np.asarray(matrix, np.float64)

    if matrix.ndim != 2:
        return None

    if not isinstance(axis, int):
            if axis is not None:
                return None

    if axis is not None and (axis >= matrix.ndim or axis < -matrix.ndim):
            return None   
    
    if norm_type=="l1":
        m = np.linalg.norm(matrix, ord=1, axis=axis, keepdims=True)
        
    elif norm_type=="l2":
        m = np.linalg.norm(matrix, ord='fro' if axis is None else 2, axis=axis, keepdims=True)
        
    elif norm_type=="max":
        m = np.linalg.norm(matrix, ord=np.inf, axis=axis, keepdims=True)

    else:
        return None
    

    normalized_matrix = np.divide(matrix, m, out=np.zeros_like(matrix), where= m!=0)

    return normalized_matrix

    
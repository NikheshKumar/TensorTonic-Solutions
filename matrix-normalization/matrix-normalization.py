import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    """
    Normalize a 2D matrix along specified axis using specified norm.
    """
    # Write code here

    matrix = np.asarray(matrix, dtype=float)

    if matrix.ndim != 2:
        return None

    if axis is not None:
        if not isinstance(axis, int):
            return None
        if axis >= matrix.ndim or axis < -matrix.ndim:
            return None    

    if norm_type == 'l2':
        ans = np.sqrt( np.sum(matrix**2, axis=axis, keepdims=True) )
        
    elif norm_type == 'l1':
        ans = np.sum(np.abs(matrix), axis=axis, keepdims=True)
        
    elif norm_type == 'max':  
        ans = np.max(np.abs(matrix), axis=axis, keepdims=True) 

    else:
        return None
 

    ans[ans == 0] = 1     
        
    normalized_matrix = matrix / ans
    

    return np.atleast_1d(normalized_matrix)    


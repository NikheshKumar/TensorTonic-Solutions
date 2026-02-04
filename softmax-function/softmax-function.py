import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    # Write code here
    x = np.asarray(x, float)

    if x.ndim==1:
        ans = np.exp(x- np.max(x)) / np.sum( np.exp(x- np.max(x)) )
    elif x.ndim==2:
        ans = np.exp(x- np.max(x, axis=1, keepdims=True)) / np.sum( np.exp(x- np.max(x, axis=1, keepdims=True)), axis=1, keepdims=True )   

    return ans
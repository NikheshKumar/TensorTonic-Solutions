import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    # Write code here
    x = np.asarray(x, float)
  
    m = np.max(x, axis=-1, keepdims=True)
  
    num = np.exp(x-m)
    den = np.sum(np.exp(x-m), axis=-1, keepdims=True)

    ans = num/den
  
    return ans
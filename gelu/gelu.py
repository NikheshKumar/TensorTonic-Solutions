import numpy as np
import math

def gelu(x):
    """
    Compute the Gaussian Error Linear Unit (exact version using erf).
    x: scalar, list, or np.ndarray
    Return: np.ndarray of same shape (dtype=float)
    """
    # Write code here
    x = np.asarray(x)
    k = np.sqrt(2)
    l = np.vectorize(math.erf)((x/k))
    g = x*(1+l)/2
    return g
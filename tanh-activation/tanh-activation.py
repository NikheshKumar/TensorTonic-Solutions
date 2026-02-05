import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    # Write code here
    x = np.asarray(x, float)

    y = np.clip(x,-20,20)

    num = np.exp(y) - np.exp(-y)
    den = np.exp(y) + np.exp(-y)

    ans = num / den

    return np.atleast_1d(ans)
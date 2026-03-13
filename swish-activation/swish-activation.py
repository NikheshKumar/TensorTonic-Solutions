import numpy as np

def swish(x):
    """
    Implement Swish activation function.
    """
    # Write code here
    x = np.atleast_1d(np.array(x, float))

    x_clip = np.clip(x, -100, 100)

    sig = np.exp(x_clip) / (1+np.exp(x_clip))

    return x_clip * sig
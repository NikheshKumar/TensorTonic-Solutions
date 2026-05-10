def xavier_initialization(W, fan_in, fan_out):
    """
    Scale raw weights to Xavier uniform initialization.
    """
    # Write code here
    import numpy as np 

    W = np.asarray(W)

    L = np.sqrt(6/(fan_in + fan_out))

    W_new = 2*L*W - L

    return W_new
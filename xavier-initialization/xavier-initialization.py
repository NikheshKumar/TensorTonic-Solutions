def xavier_initialization(W, fan_in, fan_out):
    """
    Scale raw weights to Xavier uniform initialization.
    """
    # Write code here
    import numpy as np 

    W = np.asarray(W, float)
    limit = np.sqrt(6 / (fan_in + fan_out) )

    W_new = W*2*limit - limit

    return W_new.tolist()
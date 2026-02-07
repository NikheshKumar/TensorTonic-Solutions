def he_initialization(W, fan_in):
    """
    Scale raw weights to He uniform initialization.
    """
    # Write code here
    import numpy as np 

    W = np.asarray(W, float)
    limit = np.sqrt(6/fan_in)

    W_new = W*2*limit - limit

    return W_new.tolist()

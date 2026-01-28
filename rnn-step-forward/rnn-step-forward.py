import numpy as np

def rnn_step_forward(x_t, h_prev, Wx, Wh, b):
    """
    Returns: h_t of shape (H,)
    """
    # Write code here
    x_t = np.asarray(x_t)
    h_prev = np.asarray(h_prev)
    
    h_t = np.tanh((x_t @ Wx) + (h_prev @ Wh) + b)
    h_t = np.asarray(h_t)
    return h_t

import numpy as np

def rnn_step_forward(x_t, h_prev, Wx, Wh, b):
    """
    Returns: h_t of shape (H,)
    """
    # Write code here
    
    x_t = np.asarray(x_t, float)
    h_prev = np.asarray(h_prev, float)
    Wx = np.asarray(Wx, float)
    Wh = np.asarray(Wh, float)
    b = np.asarray(b, float)


    h_t = np.tanh(x_t @ Wx + h_prev @ Wh + b )

    return h_t

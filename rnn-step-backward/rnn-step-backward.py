import numpy as np

def rnn_step_backward(dh, cache):
    """
    Returns:
        dx_t: gradient wrt input x_t      (shape: D,)
        dh_prev: gradient wrt previous h (shape: H,)
        dW: gradient wrt W               (shape: H x D)
        dU: gradient wrt U               (shape: H x H)
        db: gradient wrt bias            (shape: H,)
    """
    # Write code here

    x_t, h_prev, h_t, W, U, b = cache

    dh = np.asarray(dh, float)
    x_t = np.asarray(x_t, float)
    h_prev = np.asarray(h_prev, float)
    W = np.asarray(W, float)
    U = np.asarray(U, float)
    h_t = np.asarray(h_t, float)

    dz_t = dh * (1.0 - h_t**2)

    dx_t   = np.dot(W.T, dz_t)        
    dh_prev = np.dot(U.T, dz_t)      

    dW = np.outer(dz_t, x_t)          
    dU = np.outer(dz_t, h_prev)       

    db =  dz_t           
    return dx_t, dh_prev, dW, dU, db
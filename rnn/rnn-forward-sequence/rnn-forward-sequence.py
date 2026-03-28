import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray,
                W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> tuple:
    """
    Forward pass through entire sequence.
    """
    # YOUR CODE HERE
    B,T,N = X.shape
    h_prev = h_0
    h_list = []

    for t in range(T):
        x_t = X[:,t,:]

        h = np.tanh(h_prev@W_hh.T + x_t@W_xh.T + b_h)
        
        h_prev = h

        h_list.append(h_prev)

    h_list = np.stack(h_list, axis=1)

    return h_list, h_prev
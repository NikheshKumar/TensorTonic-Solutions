import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray,
                W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> tuple:
    """
    Forward pass through entire sequence.
    """
    # YOUR CODE HERE

    batch_size, T, input_dim = X.shape
    hidden_dim = h_0.shape[1]
    hidden_states = []
    h = h_0
  
    for t in range(T):
      x_t = X[:, t, :]
      h = np.tanh( x_t @ W_xh.T + h @ W_hh + b_h)
      hidden_states.append(h)

    hidden_states = np.stack(hidden_states, axis=1)

    return hidden_states, h
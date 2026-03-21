import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def gru_cell(x_t: np.ndarray, h_prev: np.ndarray,
             W_r: np.ndarray, W_z: np.ndarray, W_h: np.ndarray,
             b_r: np.ndarray, b_z: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    """Complete GRU cell forward pass."""
    # YOUR CODE HERE
    
    r_t = sigmoid(np.concatenate([h_prev, x_t], axis=-1) @ W_r.T + b_r)

    z_t = sigmoid(np.concatenate([h_prev, x_t], axis=-1) @ W_z.T + b_z)

    h_tilde = np.tanh(np.concatenate([r_t * h_prev, x_t], axis=-1) @ W_h.T + b_h)

    h_t = z_t * h_prev + (1-z_t)*h_tilde

    return h_t
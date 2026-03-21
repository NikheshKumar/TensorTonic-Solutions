import numpy as np

def candidate_hidden(h_prev: np.ndarray, x_t: np.ndarray, r_t: np.ndarray,
                     W_h: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    """Compute candidate: h_tilde = tanh(W_h @ [r*h, x] + b_h)"""
    # YOUR CODE HERE

    reset_gate_h = r_t * h_prev
    h_tilde = np.tanh(np.concatenate([reset_gate_h, x_t], axis=-1) @ W_h.T + b_h)

    return h_tilde
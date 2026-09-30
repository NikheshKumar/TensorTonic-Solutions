import numpy as np

def apply_causal_mask(scores: list, mask_value: float = -1e9) -> np.ndarray:
    """
    Returns a causally masked NumPy array matching the shape of scores.
    """
    # Write code here
    scores = np.asarray(scores, dtype=np.float64)
    seq_q, seq_k = scores.shape[-2], scores.shape[-1]

    mask = np.triu(np.ones((seq_q, seq_k), dtype=bool), 1)

    return np.where(mask, mask_value, scores)
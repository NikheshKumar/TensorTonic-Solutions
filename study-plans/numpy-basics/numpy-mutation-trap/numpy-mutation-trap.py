import numpy as np

def original_and_clipped(data, row_idx, lo, hi):
    """
    Returns: 2D ndarray of float64 with shape (2, ncols)
    """
    data = np.asarray(data, dtype=np.float64)

    original_row = data[row_idx, :]

    mask = np.clip(original_row, lo, hi)

    res = np.stack([original_row, mask], axis=0)

    return res.astype(np.float64)

    
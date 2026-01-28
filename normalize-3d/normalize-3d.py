import numpy as np

def normalize_3d(v):
    """
    Normalize 3D vector(s) to unit length.
    """
    # Your code here
    v = np.asarray(v, dtype=float)

    f = np.linalg.norm(v, axis=-1, keepdims=True)

    v_normalized = v.copy()

    mask = f > 1e-10

    np.divide(v, f, out=v_normalized, where=mask)

    return v_normalized
import numpy as np

def normalize_3d(v):
    """
    Normalize 3D vector(s) to unit length.
    """
    # Your code here
    v = np.asarray(v, float)
    
    v_norm = np.linalg.norm(v,ord=2, axis=-1, keepdims=True)
    
    mask = v_norm >1e-8
    
    np.divide(v, v_norm, where=mask, out=v)
    
    return v

    
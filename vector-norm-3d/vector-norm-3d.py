import numpy as np

def vector_norm_3d(v):
    """
    Compute the Euclidean norm of 3D vector(s).
    """
    # Your code here

    v = np.asarray(v, float)

    if v.ndim==1:
        return np.linalg.norm(v)
    else:
        ans = np.linalg.norm(v, axis=1)
        return ans 
    
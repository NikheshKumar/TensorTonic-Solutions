import numpy as np

def clip_gradients(g, max_norm):
    """
    Clip gradients using global norm clipping.
    """
    # Write code here
    g = np.asarray(g)
    l2_norm = np.linalg.norm(g)

    if max_norm<=0.0:
        return g
    if l2_norm==0.0:
        return g

    if l2_norm <= max_norm:
        return g    

    return g * max_norm / l2_norm        
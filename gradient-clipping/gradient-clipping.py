import numpy as np

def clip_gradients(g, max_norm):
    """
    Clip gradients using global norm clipping.
    """
    # Write code here
    g = np.asarray(g, dtype=np.float64)

    g_norm = np.sqrt(np.sum(g * g))

    if max_norm <= 0 or g_norm==0.0:
        return g

    if g_norm > max_norm and g_norm != 0:
        return g * (max_norm / g_norm)

    return g
        
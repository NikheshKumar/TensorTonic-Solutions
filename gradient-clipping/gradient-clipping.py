import numpy as np

def clip_gradients(g, max_norm):
    """
    Clip gradients using global norm clipping.
    """
    # Write code here
    g = np.asarray(g, float)
    g_norm = np.linalg.norm(g)
    
    if max_norm<=0 or g_norm==0:
      return g
    elif g_norm <= max_norm:
      return g
    else:
      return g*max_norm / g_norm
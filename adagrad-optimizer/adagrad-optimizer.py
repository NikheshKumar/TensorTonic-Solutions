import numpy as np

def adagrad_step(w, g, G, lr=0.01, eps=1e-8):
    """
    Perform one AdaGrad update step.
    """
    # Write code here
    w = np.asarray(w, np.float64)
    g = np.asarray(g, np.float64)
    G = np.asarray(G, np.float64)

    new_G = G + g**2

    new_w = w - lr*g / np.sqrt(new_G + eps)

    return new_w, new_G
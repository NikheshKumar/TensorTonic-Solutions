import numpy as np

def adagrad_step(w, g, G, lr=0.01, eps=1e-8):
    """
    Perform one AdaGrad update step.
    """
    # Write code here

    w = np.asarray(w, float)
    g = np.asarray(g, float)
    G = np.asarray(G, float)

    G_new = G + g**2

    w_new = w - ((lr * g) / (np.sqrt(G_new + eps)))

    return w_new, G_new
    
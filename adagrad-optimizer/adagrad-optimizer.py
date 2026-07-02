import numpy as np

def adagrad_step(w, g, G, lr=0.01, eps=1e-8):
    """
    Perform one AdaGrad update step.
    """
    # Write code here
    g = np.asarray(g, np.float64)
    G = np.asarray(G, np.float64)
    w = np.asarray(w, np.float64)

    G = G + g**2

    w = w - lr*g/(np.sqrt(G + eps))


    return w, G
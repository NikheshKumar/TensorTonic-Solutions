import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    """
    # Write code here
    w = np.asarray(w) 
    g = np.asarray(g) 
    s = np.asarray(s) 

    new_s = beta * s + (1 - beta) * (g * g) 
    new_w = w - (lr * g) / (np.sqrt(new_s + eps)) 
    
    return new_w.tolist(), new_s.tolist()
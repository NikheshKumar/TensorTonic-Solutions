import numpy as np

def adamw_step(w, m, v, grad, lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.01, eps=1e-8):
    """
    Perform one AdamW update step.
    """
    # Write code here
    w = np.asarray(w, np.float64)
    m = np.asarray(m, np.float64)
    v = np.asarray(v, np.float64)
    grad = np.asarray(grad, np.float64)

    m = beta1*m + (1-beta1)*grad
    
    v = beta2*v + (1-beta2)*(grad**2)

    w = w - lr*(weight_decay*w) - lr*m/(np.sqrt(v)+eps)

    return w, m, v
    
import numpy as np

def nesterov_momentum_step(w, v, grad, lr=0.01, momentum=0.9):
    """
    Perform one Nesterov Momentum update step.
    """
    # Write code here
    w = np.asarray(w)
    v = np.asarray(v)
    grad = np.asarray(grad)

    w_look = w - momentum * v

    v = momentum * v + lr * grad

    w = w - v

    return w, v
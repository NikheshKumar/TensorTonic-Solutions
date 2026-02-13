import numpy as np

def adadelta_step(w, grad, E_grad_sq, E_update_sq, rho=0.9, eps=1e-6):
    """
    Perform one AdaDelta update step.
    """
    # Write code here
    w = np.asarray(w)
    grad = np.asarray(grad)
    E_grad_sq = np.asarray(E_grad_sq)
    E_update_sq = np.asarray(E_update_sq)

    E_grad_sq = rho*E_grad_sq + (1-rho)*(grad**2)

    num = -np.sqrt( E_update_sq + eps )
    den = np.sqrt( E_grad_sq + eps )

    w_diff = num * grad / den

    E_update_sq = rho*E_update_sq + (1-rho)*(w_diff**2)

    w = w + w_diff

    return w, E_grad_sq, E_update_sq
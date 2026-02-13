import numpy as np

def kl_divergence(p, q, eps=1e-12):
    """
    Compute KL Divergence D_KL(P || Q).
    """
    # Write code here
    p = np.asarray(p, float)
    q = np.asarray(q, float)

    q_stable = q + eps

    mask = p>0
    p_new = p[mask]
    q_stable = q_stable[mask]

    d = np.sum(p_new* np.log(p_new/q_stable) )
    return d
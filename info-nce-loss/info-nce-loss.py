import numpy as np

def info_nce_loss(Z1, Z2, temperature=0.1):
    """
    Compute InfoNCE Loss for contrastive learning.
    """
    # Write code here
    Z1 = np.asarray(Z1)
    Z2 = np.asarray(Z2)

    S = np.matmul(Z1, Z2.T) / temperature

    S_stable = S - np.max(S)

    num = np.exp(np.diag(S_stable))

    den = np.sum(np.exp(S_stable), axis=1)

    L = -np.mean(np.log(num/den), axis=0)

    return float(L)
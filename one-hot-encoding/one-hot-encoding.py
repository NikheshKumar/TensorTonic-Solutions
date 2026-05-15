import numpy as np

def one_hot(y, num_classes=None):
    """
    Convert integer labels y ∈ {0,...,K-1} into one-hot matrix of shape (N, K).
    """
    # Write code here
    y = np.asarray(y, dtype=int)
    
    if num_classes is None:
        cla, counts = np.unique(y, return_counts=True)
        num_classes = len(cla)

    N = y.shape[0]

    ohm = np.zeros((N,num_classes), dtype=int)

    i = np.arange(0,N)

    ohm[i,y] = 1

    return ohm

    

    
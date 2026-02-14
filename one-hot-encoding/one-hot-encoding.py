import numpy as np

def one_hot(y, num_classes=None):
    """
    Convert integer labels y ∈ {0,...,K-1} into one-hot matrix of shape (N, K).
    """
    # Write code here
    y = np.asarray(y, int)
  
    if num_classes is None:
      num_classes = np.max(y) + 1 if len(y) > 0 else 0

    ohm = np.zeros((len(y), num_classes), int)

    ohm[np.arange(len(y)), y]=1.0

    return ohm
      
    
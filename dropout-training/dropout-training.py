import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code here
    x = np.asarray(x, float)
  
    if rng is None:
      rng = np.random

    random_vals = rng.random(x.shape)
    mask = random_vals/(1-p) < 1

    if p==1.0:
      dropout_pattern = np.zeros_like(x)
    if p==0.0:
      dropout_pattern = np.ones_like(x)
    else:
      dropout_pattern = mask.astype(float) /(1-p)

    output = x * dropout_pattern

    return output, dropout_pattern
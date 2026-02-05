import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Compute mean, median, and mode.
    """
    # Write code here
    x = np.asarray(x, float)

    m1 = np.mean(x)
    m2 = np.median(x)
    vals, counts = np.unique(x, return_counts=True)
    m3 = np.min (vals[np.argmax(counts)])

    return m1, m2, m3



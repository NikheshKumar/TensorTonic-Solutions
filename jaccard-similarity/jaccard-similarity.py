def jaccard_similarity(set_a, set_b):
    """
    Compute the Jaccard similarity between two item sets.
    """
    # Write code here
    import numpy as np 

    j = len(np.intersect1d(set_a, set_b)) / len(np.union1d(set_a, set_b)) if len(np.union1d(set_a, set_b)) != 0 else 0.0

    return j
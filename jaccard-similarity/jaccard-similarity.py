def jaccard_similarity(set_a, set_b):
    """
    Compute the Jaccard similarity between two item sets.
    """
    # Write code here
    import numpy as np 

    i = set(set_a) & set(set_b)
    u = set(set_a) | set(set_b)

    j = len(i) / len(u) if len(u)>0 else 0.0

    return j
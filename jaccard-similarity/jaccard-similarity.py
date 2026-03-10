def jaccard_similarity(set_a, set_b):
    """
    Compute the Jaccard similarity between two item sets.
    """
    # Write code here
    import numpy as np 

    seta = np.asarray(set_a)
    setb = np.asarray(set_b)

    j = len(np.intersect1d(seta, setb)) / len(np.union1d(seta, setb)) if len(np.union1d(seta, setb))!=0 else 0.0

    return j
def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    # Write code here
    import numpy as np 

    x1, x2 = np.asarray(x1, float), np.asarray(x2, float)

    cosine = np.dot(x1, x2) / ( np.linalg.norm(x1) * np.linalg.norm(x2) )

    if label == 1:
        return 1 - cosine

    else:
        return max( 0, cosine-margin )    
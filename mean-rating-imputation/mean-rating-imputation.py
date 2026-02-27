def mean_rating_imputation(ratings_matrix, mode):
    """
    Fill missing ratings (zeros) with user or item means.
    """
    # Write code here
    import numpy as np 

    ratings_matrix = np.asarray(ratings_matrix, float)

    axis = 1 if mode == 'user' else 0

    mask = ratings_matrix > 0

    tot = np.sum(ratings_matrix, axis=axis)
    counts = np.sum(mask, axis=axis)

    m = np.divide(tot, counts, out=np.zeros_like(tot, float), where=counts != 0)

    if mode=='user':
      vals = m[:, np.newaxis]
    if mode=='item':
      vals = m

    ratings_new = np.where(ratings_matrix == 0, vals, ratings_matrix)

    return ratings_new.tolist()
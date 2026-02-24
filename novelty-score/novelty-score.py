def novelty_score(recommendations, item_counts, n_users):
    """
    Compute the average novelty of a recommendation list.
    """
    # Write code here
    import numpy as np 

    recommendations = np.asarray(recommendations)
    item_counts = np.asarray(item_counts)
    N = len(recommendations)

    novelty = ( -np.sum( np.log2(item_counts / n_users) ) ) / N

    return novelty
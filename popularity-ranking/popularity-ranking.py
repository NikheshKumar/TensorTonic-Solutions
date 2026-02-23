def popularity_ranking(items, min_votes, global_mean):
    """
    Compute the Bayesian weighted rating for each item.
    """
    # Write code here
    import numpy as np 

    items = np.asarray(items)

    wr = (items[:,1]*items[:,0] / (items[:,1] + min_votes) ) + ( min_votes*global_mean / (items[:,1]+min_votes) )

    return wr.tolist()
    
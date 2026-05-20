def hit_rate_at_k(recommendations, ground_truth, k):
    """
    Compute the hit rate at K.
    """
    # Write code here
    import numpy as np 


    recommendations = np.asarray(recommendations, int)
    ground_truth = np.asarray(ground_truth, int)

    tot = 0

    def f_count(arr, g):
        return np.sum(arr==g)
        

    for user, g in zip(recommendations, ground_truth):
        tot += f_count(user[:k], g)

    return tot / len(recommendations)
def hit_rate_at_k(recommendations, ground_truth, k):
    """
    Compute the hit rate at K.
    """
    # Write code here
    import numpy as np 

    recommendations = np.asarray(recommendations)
    ground_truth = np.asarray(ground_truth)

    topk = recommendations[:, :k]
    matches = (ground_truth==topk)

    count = np.any(matches, axis=1)

    hr = np.mean(count.astype(float))

    return hr
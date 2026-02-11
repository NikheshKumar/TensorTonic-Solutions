def hit_rate_at_k(recommendations, ground_truth, k):
    """
    Compute the hit rate at K.
    """
    # Write code here
    import numpy as np 

    recommendations = np.asarray(recommendations, object)
    ground_truth = np.asarray(ground_truth, object)

    matches = []

    for gt, r in zip(ground_truth, recommendations):
        topk = r[:k]
        m = len(set(topk).intersection(set(gt))) > 0
        matches.append(m)

    return float(np.mean(matches))
def adjusted_cosine_similarity(ratings_matrix, item_i, item_j):
    """
    Compute adjusted cosine similarity between two items.
    """
    # Write code here
    import numpy as np 

    ratings_matrix = np.asarray(ratings_matrix, dtype=np.float64)

    mask = ratings_matrix > 0

    user_ratings_sum = np.sum(ratings_matrix, axis=1)
    user_counts = np.sum(mask, axis=1)

    r_mean = np.divide(user_ratings_sum, user_counts, out=np.zeros_like(user_ratings_sum, dtype=np.float64), where=user_counts > 0)

    r_ui = ratings_matrix[:, item_i]
    r_uj = ratings_matrix[:, item_j]
    common_U = (r_ui > 0) & (r_uj > 0)

    if not np.any(common_U):
        return 0.0

    num = np.sum( (r_ui[common_U] - r_mean[common_U])*(r_uj[common_U] - r_mean[common_U]) )

    den = np.sqrt(np.sum((r_ui[common_U] - r_mean[common_U])**2)) * np.sqrt(np.sum((r_uj[common_U] - r_mean[common_U])**2))

    return num/den if den > 1e-4 else 0.0

    
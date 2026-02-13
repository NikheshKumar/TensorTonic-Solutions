def baseline_predict(ratings_matrix, target_pairs):
    """
    Compute baseline predictions using global mean and user/item biases.
    """
    # Write code here
    import numpy as np 

    ratings_matrix = np.asarray(ratings_matrix)

    mask = ratings_matrix > 0
    if not np.any(mask):
        return [0.0] * len(target_pairs)

    mu = np.mean(ratings_matrix[mask])

    bi = np.zeros(ratings_matrix.shape[1])
    for i in range(len(bi)):
        item_ratings = ratings_matrix[mask[:, i], i]  
        if item_ratings.size > 0:
            bi[i] = np.mean(item_ratings) - mu   

    bu = np.zeros(ratings_matrix.shape[0])
    for j in range(len(bu)):
        user_ratings = ratings_matrix[j, mask[j, :]]
        if user_ratings.size > 0:
            bu[j] = np.mean(user_ratings) - mu   

    bp_matrix = []

    for target in target_pairs:
        bp = mu + bu[target[0]] + bi[target[1]]
        bp_matrix.append(bp)

    return bp_matrix    

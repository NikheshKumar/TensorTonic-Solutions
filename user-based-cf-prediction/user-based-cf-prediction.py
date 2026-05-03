def user_based_cf_prediction(similarities, ratings):
    """
    Predict a rating using user-based collaborative filtering.
    """
    # Write code here
    import numpy as np 

    similarities = np.asarray(similarities)
    ratings = np.asarray(ratings)

    mask = similarities > 0.0

    num = np.sum(similarities[mask]*ratings[mask])

    den = np.sum(similarities[mask])

    return num/den if den>0.0 else 0.0
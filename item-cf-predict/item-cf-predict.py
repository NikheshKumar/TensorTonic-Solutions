def item_cf_predict(user_ratings, item_similarities, target):
    """
    Predict the rating using item-based collaborative filtering.
    """
      # Write code here
    import numpy as np 

    user_ratings = np.asarray(user_ratings)
    item_similarities = np.asarray(item_similarities)

    mask = (user_ratings>0) & (item_similarities>0)
    mask[target] = False

    num = np.sum(item_similarities[mask] * user_ratings[mask])

    den = np.sum(item_similarities[mask])

    return num/den if den>0 else 0.0
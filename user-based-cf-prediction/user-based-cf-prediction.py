def user_based_cf_prediction(similarities: list, ratings: list) -> float:
    """
    Returns the positive-similarity weighted rating prediction.
    """
    # Write code here
    num = den = 0
    
    for s, r in zip(similarities, ratings):
        if s>0:
            num += s * r
            den += s


    return num / den if den!=0 else 0
        
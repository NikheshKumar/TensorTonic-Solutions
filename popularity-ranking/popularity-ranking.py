def popularity_ranking(items: list, min_votes: int, global_mean: float) -> list:
    """
    Returns the weighted rating for every item.
    """
    # Write code here
    wr = []
    c = global_mean
    m = min_votes

    for r, v in items:
        val = (v*r + m*c)/(v+m)
        wr.append(val)

    return wr
        
        
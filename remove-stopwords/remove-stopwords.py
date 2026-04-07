def remove_stopwords(tokens, stopwords):
    """
    Returns: list[str] - tokens with stopwords removed (preserve order)
    """
    # Your code here
    import numpy as np 

    stop_set = set(stopwords)
    res = []

    for t in tokens:
        if t not in stop_set:
            res.append(t)

    return res
        
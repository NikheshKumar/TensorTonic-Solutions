def remove_stopwords(tokens, stopwords):
    """
    Returns: list[str] - tokens with stopwords removed (preserve order)
    """
    # Your code here
    stop = set(stopwords)
    ans = []

    for token in tokens:
        if token in stop:
            continue
        else:
            ans.append(token)    

    return ans        
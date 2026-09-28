def cumulative_returns(returns: list) -> list:
    """
    Returns the compounded cumulative return after every period.
    """
    # Write code here
    w = 1.0

    re = []

    for i in range(len(returns)):
        w = w*(1.0 + returns[i])
        re.append(w-1.0)

    return re
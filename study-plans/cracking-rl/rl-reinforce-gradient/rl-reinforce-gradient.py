def reinforce_loss(log_probs: list, returns: list) -> float:
    """
    Returns the REINFORCE loss as a float rounded to four decimals.
    """
    l = -sum(l*g for l,g in zip(log_probs, returns))
    l = l / len(log_probs)
    return round(float(l),4)
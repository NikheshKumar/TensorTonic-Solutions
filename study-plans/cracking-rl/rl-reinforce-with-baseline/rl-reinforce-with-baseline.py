def reinforce_baseline_loss(log_probs: list, returns: list, baselines: list) -> float:
    """
    Returns the baseline-adjusted loss as a float rounded to four decimals.
    """
    l = -sum(l*(g-b) for l,g,b in zip(log_probs,returns,baselines))
    l = float(l / len(log_probs))
    return round(l,4)
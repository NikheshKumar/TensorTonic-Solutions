def per_priorities_and_weights(td_errors: list, alpha: float, beta: float, epsilon: float = 1e-6) -> tuple:
    """
    Returns (probs, is_weights), two float lists rounded to four decimals.
    """
    N = len(td_errors)
    if N==0:
        return [], []


    priorities = [float((abs(t) + epsilon) ** alpha) for t in td_errors]
    probs = [float(p/sum(priorities)) for p in priorities]
    weights = [float((N*p/sum(probs))**(-beta)) for p in probs]
    is_weights = [round(float(w/max(weights)),4) for w in weights]

    
    return [round(float(p),4) for p in probs], is_weights
        
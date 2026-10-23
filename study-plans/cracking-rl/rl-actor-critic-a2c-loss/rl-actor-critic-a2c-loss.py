def a2c_loss(log_probs: list, advantages: list, values: list, returns: list, entropies: list, value_coef: float = 0.5, entropy_coef: float = 0.01) -> float:
    """
    Returns the total A2C loss as a float rounded to four decimals.
    """
    T = len(log_probs)

    l_pi = -sum(l*a/T for l,a in zip(log_probs,advantages))

    l_v = sum((g-v)**2/T for g,v in zip(returns, values))

    loss = l_pi + value_coef * l_v - entropy_coef * sum(entropies)/len(entropies)

    loss = round(float(loss),4)

    return loss
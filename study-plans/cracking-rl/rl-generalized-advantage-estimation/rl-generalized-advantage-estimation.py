def gae_advantages(rewards: list, values: list, gamma: float, lam: float, last_value: float = 0.0) -> list:
    """
    Returns the advantage list in timestep order, rounded to four decimals.
    """
    values.append(last_value)
    T = len(rewards)
    delta = [rewards[t] + gamma*values[t+1]-values[t] for t in range(T)]

    A = [0]*T
    a = 0.0
    for t in reversed(range(T)):
        a = delta[t] + gamma * lam * a
        A[t] = a

    return [round(float(a),4) for a in A]
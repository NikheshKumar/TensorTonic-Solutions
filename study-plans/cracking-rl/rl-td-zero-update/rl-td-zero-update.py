def td_zero_update(V: list, transitions: list, alpha: float, gamma: float) -> list:
    """
    Returns the updated state-value list, rounded to four decimals.
    """

    for s, r, s_next in transitions:
        V[s] = V[s] + alpha * (r + gamma * V[s_next] - V[s])

    V = [round(float(v), 4) for v in V]

    return V
    
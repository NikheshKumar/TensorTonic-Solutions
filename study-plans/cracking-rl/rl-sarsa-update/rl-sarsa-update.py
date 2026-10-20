def sarsa_update(Q: list, transitions: list, alpha: float, gamma: float) -> list:
    """
    Returns the updated Q-table as a nested list, rounded to four decimals.
    """

    for s, a, reward, s_next, a_next in transitions:
        Q[s][a] = Q[s][a] + alpha * (reward + gamma * Q[s_next][a_next] - Q[s][a])

    return [[round(float(v),4) for v in rows] for rows in Q]
    
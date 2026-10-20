def expected_sarsa_update(Q: list, transitions: list, policy: list, alpha: float, gamma: float) -> list:
    """
    Returns a new updated Q-table as a nested list, rounded to four decimals.
    """
    Q_new = [row[:] for row in Q]

    for s, a, reward, s_next in transitions:
        pwa = sum(pi * q for pi,q in zip(policy[s_next], Q_new[s_next]))
        Q_new[s][a] = Q_new[s][a] + alpha * (reward + gamma*pwa - Q_new[s][a])

    return [[round(float(q), 4) for q in rows] for rows in Q_new]
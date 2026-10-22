def double_dqn_targets(Q_online_next: list, Q_target_next: list, rewards: list, dones: list, gamma: float) -> list:
    """
    Returns one Double DQN target per sample, rounded to four decimals.
    """
    Y = []
    
    for i in range(len(rewards)):
        a_star = Q_online_next[i].index(max(Q_online_next[i]))

        y = rewards[i] + gamma*(1-dones[i])*Q_target_next[i][a_star]

        Y.append(y)

    return [round(float(y),4) for y in Y]
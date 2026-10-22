def dqn_loss(Q_online: list, Q_target_next: list, actions: list, rewards: list, dones: list, gamma: float) -> float:
    """
    Returns the mean squared TD loss, rounded to four decimals.
    """
    B = len(rewards)
    l = 0

    if B==0:
        return 0.0
    
    for i in range(B):
        y = rewards[i] + gamma * (1.0 - float(dones[i])) * max(Q_target_next[i])
        l += (y - Q_online[i][actions[i]])**2

    return round(float(l/B),4)
        
def gae(rewards, values, gamma, lam):
    """
    Compute Generalized Advantage Estimation.
    """
    # Write code here
    import numpy as np

    rewards = np.asarray(rewards, float)
    values = np.asarray(values, float)

    T = len(rewards)
    delta = np.zeros(T, float)
    adv = np.zeros(T, float)

    adv[T-1] = delta[T-1]
    last_adv = 0

    for t in range(T-1, -1, -1):
        delta[t] = rewards[t] + gamma * values[t+1] - values[t]
        adv[t] = delta[t] + gamma * lam * last_adv
        last_adv = adv[t]

    return adv.tolist()

    
        
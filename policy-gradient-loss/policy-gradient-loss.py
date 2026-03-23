def policy_gradient_loss(log_probs, rewards, gamma):
    """
    Compute REINFORCE policy gradient loss with mean-return baseline.
    """
    # Write code here
    import numpy as np 
    
    log_probs = np.asarray(log_probs, float)
    rewards = np.asarray(rewards, float)

    T = len(rewards)
    if T == 0:
        return 0.0
    
    G = np.zeros(T, float)
    
    G[T-1] = rewards[T-1]

    for t in range(T-2, -1, -1):
        G[t] = rewards[t] + gamma * G[t+1]

    mean_G = np.mean(G)

    adv = [g - mean_G for g in G]

    loss = -np.mean(log_probs * adv)

    return float(loss)

    
        
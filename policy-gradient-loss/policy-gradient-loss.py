def policy_gradient_loss(log_probs, rewards, gamma):
    """
    Compute REINFORCE policy gradient loss with mean-return baseline.
    """
    # Write code here
    import numpy as np 

    log_probs = np.asarray(log_probs, np.float64)
    rewards = np.asarray(rewards, np.float64)

    T = len(rewards)
    
    if T==0:
        return 0.0

    G = np.zeros((T,), np.float64)
    
    G[-1] = rewards[-1].astype(np.float64)

    for t in range(T-2, -1, -1):
        G[t] = rewards[t] + gamma*G[t+1]

    mean_G = np.mean(G)

    adv = G - mean_G

    loss = -np.mean(log_probs * adv)

    return float(loss)
    
    
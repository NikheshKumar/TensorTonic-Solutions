def rlhf_ppo_kl_loss(log_probs_new, log_probs_old, log_probs_ref, advantages, clip_eps, kl_coef):
    """
    Returns: float, RLHF PPO loss with KL penalty rounded to 4 decimals
    """
    import math

    A = advantages
    T = len(advantages)
    loss = 0.0

    for t in range(T):
        r = math.exp(log_probs_new[t] - log_probs_old[t])
        clip_r = max(1-clip_eps, min(1+clip_eps, r))
        loss  += -min(r*A[t], clip_r*A[t]) + kl_coef*(log_probs_new[t] - log_probs_ref[t])

    return round(loss/T, 4)
    

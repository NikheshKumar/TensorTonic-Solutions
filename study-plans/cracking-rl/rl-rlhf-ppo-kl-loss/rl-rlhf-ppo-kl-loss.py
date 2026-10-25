import math

def rlhf_ppo_kl_loss(log_probs_new: list, log_probs_old: list, log_probs_ref: list, advantages: list, clip_eps: float, kl_coef: float) -> float:
    """
    Returns the combined PPO and reference-penalty loss, rounded to four decimals.
    """
    def clip(z, a, b):
        if z<a:
            return a
        elif z>b:
            return b
        else:
            return z
    
    r = [math.exp(l1-l2) for l1,l2 in zip(log_probs_new, log_probs_old)]

    T = len(r)

    l_clip = -sum(min(r[t]*advantages[t], clip(r[t], 1-clip_eps, 1+clip_eps)*advantages[t]) for t in range(T))
    l_clip = l_clip / T

    K = sum(l1-l2 for l1,l2 in zip(log_probs_new, log_probs_ref))
    K = K/T

    L = l_clip + kl_coef * K

    return round(L,4)

    
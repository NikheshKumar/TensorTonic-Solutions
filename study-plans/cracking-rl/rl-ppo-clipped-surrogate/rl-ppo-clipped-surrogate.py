import math

def ppo_clipped_loss(log_probs_new: list, log_probs_old: list, advantages: list, clip_eps: float) -> float:
    """
    Returns the PPO clipped loss as a float rounded to four decimals.
    """
    def clip(z,a,b):
        if z<a:
            return a
        elif z>b:
            return b
        else:
            return z
            
    T = len(log_probs_new)
    rewards = [math.exp(log_probs_new[t] - log_probs_old[t]) for t in range(T)]

    loss = -sum(min(r*a, clip(r,1-clip_eps,1+clip_eps)*a) for r,a in zip(rewards, advantages))

    loss = round(float(loss/T),4)

    return loss
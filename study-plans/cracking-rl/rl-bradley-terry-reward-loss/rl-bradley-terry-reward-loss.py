import math

def bradley_terry_loss(r_chosen: list, r_rejected: list) -> float:
    """
    Returns the mean preference loss as a float rounded to four decimals.
    """
    def log_sigmoid(z):
        ans = min(0.0,z) - math.log1p(math.exp(-abs(z)))
        return ans

    n = len(r_chosen)
    l = -sum(log_sigmoid(rc - rej) for rc, rej in zip(r_chosen, r_rejected))
    l = l / n

    return round(l,4)
    
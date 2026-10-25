import math

def dpo_loss(log_pi_new_w: list, log_pi_new_l: list, log_pi_ref_w: list, log_pi_ref_l: list, beta: float) -> float:
    """
    Returns the mean DPO loss as a float rounded to four decimals.
    """
    def log_sigmoid(z):
        return min(0.0,z) - math.log1p(math.exp(-abs(z)))
            
    z = [beta*(l1-l2)-beta*(l3-l4) for l1,l2,l3,l4 in zip(log_pi_new_w,log_pi_ref_w,log_pi_new_l,log_pi_ref_l)]

    l = -sum(log_sigmoid(y) for y in z)
    l = l / len(z)

    return round(float(l),4)

    
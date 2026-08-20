import math

def dpo_loss(log_pi_new_w, log_pi_new_l, log_pi_ref_w, log_pi_ref_l, beta):
    """
    Returns: float, DPO loss rounded to 4 decimals
    """
    N = len(log_pi_new_l)
    loss = 0.0

    def logsig(z):
        return math.log(1.0+math.exp(-z)) if z>=0.0 else math.log(1.0+math.exp(z)) - z
        
    for n in range(N):
        term = beta * ((log_pi_new_w[n] - log_pi_ref_w[n])-(log_pi_new_l[n] - log_pi_ref_l[n]))
        loss += logsig(term)


    return round(loss/N, 4)

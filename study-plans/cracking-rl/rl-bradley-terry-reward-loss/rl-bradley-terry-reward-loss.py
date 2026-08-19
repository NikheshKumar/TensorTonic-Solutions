import math

def bradley_terry_loss(r_chosen, r_rejected):
    """
    Returns: float, Bradley-Terry preference loss rounded to 4 decimals
    """

    def logsig(x):
        return -math.log(1.0+math.exp(-x)) if x>=0.0 else x - math.log(1.0+math.exp(x))

    loss = 0.0

    for i in range(len(r_chosen)):
        loss += logsig(r_chosen[i] - r_rejected[i])

    return round(-loss/len(r_chosen), 4)

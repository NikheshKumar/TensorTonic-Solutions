from math import exp

def exponential_distribution(lam: float, t: float) -> dict:
    """
    Returns the PDF, CDF, survival probability, mean, and variance.
    """
    S = exp(-lam*t)
    f = lam * S
    F = 1 - S
    mean = 1/lam
    var = lam**(-2)

    return {"cdf":F, "mean":mean, "pdf":f, "survival":S, "variance":var}
    

import numpy as np

def linear_beta_schedule(T: int,
                         beta_1: float = 0.0001,
                         beta_T: float = 0.02) -> list[float]:
    """
    Returns T linearly spaced beta values.
    """
    beta = np.linspace(beta_1, beta_T, T, dtype=np.float64)

    return np.round(beta,6).tolist()
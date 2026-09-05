import numpy as np

def compute_ddpm_loss(epsilon: list, epsilon_pred: list) -> float:
    """
    Returns the mean DDPM noise-prediction loss.
    """
    epsilon = np.array(epsilon, dtype=np.float64)
    epsilon_pred = np.array(epsilon_pred, dtype=np.float64)
    
    l = np.mean((epsilon - epsilon_pred)**2)

    return l
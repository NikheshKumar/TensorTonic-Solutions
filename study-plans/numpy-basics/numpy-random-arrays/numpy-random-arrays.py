import numpy as np

def generate_random_array(shape, kind, seed):
    """
    Returns: 2D ndarray of float64 random values
    """
    np.random.seed(seed)
    
    if kind == "uniform":
        return np.random.uniform(low=0.0, high=1.0, size=shape).astype(np.float64)
        
    if kind == "normal":
        return np.random.standard_normal(size=shape).astype(np.float64)
        

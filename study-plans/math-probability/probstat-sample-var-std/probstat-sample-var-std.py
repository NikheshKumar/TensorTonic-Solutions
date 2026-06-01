import numpy as np

def sample_var_std(x):
    """
    Returns: dict with 'variance' and 'std_dev' as floats.
    """
    x = np.asarray(x, dtype=np.float64)

    var = np.var(x, ddof=1)
    std_dev = np.std(x, ddof=1)

    return {"variance":var, "std_dev":std_dev}
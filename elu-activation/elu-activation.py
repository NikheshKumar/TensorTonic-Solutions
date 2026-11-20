import math

def elu(x: list, alpha: float = 1.0) -> list:
    """
    Returns ELU applied elementwise to the input values.
    """
    # Write code here
    ans = [v if v>0.0 else alpha * (math.exp(v)-1.0) for v in x]
    return ans
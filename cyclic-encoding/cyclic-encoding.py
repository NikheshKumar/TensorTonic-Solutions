def cyclic_encoding(values, period):
    """
    Encode cyclic features as sin/cos pairs.
    """
    # Write code here
    import numpy as np 
    
    values = np.asarray(values, dtype=np.float64)

    theta = 2*np.pi*values/period

    return np.column_stack([np.sin(theta), np.cos(theta)]).tolist()
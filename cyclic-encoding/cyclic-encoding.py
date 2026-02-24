def cyclic_encoding(values, period):
    """
    Encode cyclic features as sin/cos pairs.
    """
    # Write code here
    import numpy as np 

    values = np.asarray(values)

    theta = 2*np.pi*values / period

    encoded = np.column_stack( (np.sin(theta), np.cos(theta)) )

    return encoded.tolist()
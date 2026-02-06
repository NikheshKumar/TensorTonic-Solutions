import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here

    pe = np.zeros((seq_len, d_model), dtype=float)

    positions = np.arange(seq_len, dtype=float).reshape(seq_len, 1)

    index = np.arange((d_model + 1) // 2, dtype=float).reshape(1, (d_model + 1) // 2) 

    den = base ** (2.0 * index / d_model)                               
    angles = positions / den                                       

    pe[:, 0::2] = np.sin(angles) 
    pe[:, 1::2] = np.cos(angles[:, : d_model // 2])     

    return pe                               
     
    

def add_positional_encoding(x, base=10000.0):
    """
    Add PE to input x of shape (B, T, d_model); return same shape.
    """
    # Write code here

    x = np.asarray(x, dtype=float)

    if x.ndim != 3:
        raise ValueError("x must have shape (B, T, d_model)")

    B, T, D = x.shape

    pe = positional_encoding(T, D, base=base)  

    return x + pe[None, :, :]  
    
import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    pe = np.zeros((seq_len, d_model), dtype=float)

    pos = np.arange(seq_len, dtype=float).reshape(seq_len, 1)

    index = np.arange((d_model + 1) // 2, dtype=float).reshape(1, (d_model + 1) // 2) 
    
    angles = pos / ( base ** (2.0 * index / d_model) )
  
    pe[:, 0::2] = np.sin(angles[:, :(d_model+1) // 2])
    pe[:, 1::2] = np.cos(angles[:, :d_model // 2])

    return pe


import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Your code here
    pe = np.zeros((seq_length, d_model))
    
    pos = np.arange(seq_length).reshape(-1,1)

    i = np.arange(0,d_model,2)

    angles = pos * np.exp(i*(-np.log(1e4))/d_model)

    pe[:,0::2] = np.sin(angles[:, :(d_model + 1) // 2])
    pe[:,1::2] = np.cos([angles[:, :d_model // 2]])

    return pe
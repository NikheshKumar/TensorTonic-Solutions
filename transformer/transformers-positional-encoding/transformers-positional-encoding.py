import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Your code here
    i = np.arange(0,d_model,2)
    den = (1e4)**(i/d_model)
    pos = np.arange(seq_length).reshape(-1,1)

    pe = np.zeros((seq_length, d_model), dtype=np.float64)

    pe[:,0::2] = np.sin(pos/den)
    pe[:,1::2] = np.cos(pos/den)

    return pe
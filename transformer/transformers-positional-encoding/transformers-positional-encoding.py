import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Your code here
    
    pe_matrix = np.zeros((seq_length, d_model), np.float64)
    pos = np.arange(seq_length).reshape(-1,1)

    pe_matrix[:,0::2] = np.sin(pos * np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)))
    
    pe_matrix[:,1::2] = np.cos(pos * np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)))


    return pe_matrix

    
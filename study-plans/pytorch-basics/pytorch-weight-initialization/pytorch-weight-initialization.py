import torch
import numpy as np

def initialize_weights(fan_in, fan_out, method):
    """
    Returns: tensor of shape (fan_out, fan_in) with initialized weights
    """
    out = torch.empty(fan_out, fan_in)
    
    if method == "xavier_uniform":
        bound = np.sqrt(6.0 / (fan_in + fan_out))
        out.uniform_(-bound, bound)
        
    elif method == "xavier_normal":
        bound = np.sqrt(2.0 / (fan_in + fan_out))
        out.normal_(0.0, bound)
        
    elif method == "he_uniform":
        bound = np.sqrt(6.0 / fan_in)
        out.uniform_(-bound, bound)
        
    elif method == "he_normal":
        bound = np.sqrt(2.0 / fan_in)
        out.normal_(0.0, bound)

    return out
    

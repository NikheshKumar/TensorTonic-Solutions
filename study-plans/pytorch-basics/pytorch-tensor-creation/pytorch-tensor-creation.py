import torch

def create_tensor(method, shape, value=0.0):
    """
    Returns: list
    """

    if method=="zeros":
        out = torch.zeros(shape, dtype=torch.float32)
        
    if method=="ones":
        out = torch.ones(shape, dtype=torch.float32)
        
    if method=="full":
        out = torch.full(shape, fill_value=value, dtype=torch.float32)
    
    return out.tolist()
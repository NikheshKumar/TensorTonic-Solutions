import torch

def reshape_tensor(x, op):
    """
    Returns: list
    """
    x = torch.as_tensor(x, dtype=torch.float32)
    out = x

    if op=="flatten":
        out = torch.flatten(x)
        
    if op=="squeeze":
        out = torch.squeeze(x)
        
    if op=="transpose":
        if x.ndim>=2:
            out = x.t()

    return out.tolist()

    

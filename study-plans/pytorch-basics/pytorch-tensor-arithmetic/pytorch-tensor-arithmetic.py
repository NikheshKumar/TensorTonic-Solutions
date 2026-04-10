import torch

def tensor_op(x, y, op):
    """
    Returns: list (result tensor converted via .tolist())
    """
    x = torch.as_tensor(x)
    y = torch.as_tensor(y)
    
    if op=="add":
        out = torch.add(x, y)

    elif op=="multiply":
        out = torch.mul(x,y)

    elif op=="matmul":
        out = torch.matmul(x,y)

    elif op=="power":
        out = torch.pow(x, y)

    elif op=="max":
        out = torch.maximum(x, y)

    return out.tolist()
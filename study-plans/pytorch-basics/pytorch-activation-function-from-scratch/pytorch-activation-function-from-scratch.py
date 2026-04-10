import torch

def activate(x, method="relu"):
    """
    Returns: list (activated tensor converted via .tolist())
    """

    x = torch.as_tensor(x, dtype=torch.float32)

    if method=="relu":
        out = torch.relu(x)
        
    if method=="sigmoid":
        out = torch.sigmoid(x)
        
    if method=="tanh":
        out = torch.tanh(x)

    if method=="leaky_relu":
        out = torch.max(x, 0.01*x)

    return out.tolist()
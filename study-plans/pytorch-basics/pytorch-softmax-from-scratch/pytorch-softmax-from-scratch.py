import torch

def softmax(logits):
    """
    Returns: tensor of same shape with softmax probabilities (each row sums to 1)
    """
    m = torch.max(logits, dim=1, keepdim=True).values
    num = torch.exp(logits - m)
    den = torch.sum(num, dim=1, keepdim=True)

    return num/den

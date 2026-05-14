import torch

def batch_norm(X, gamma, beta, eps=1e-5):
    """
    Returns: tensor of shape (N, D), the batch-normalized output
    """
    X = torch.as_tensor(X, dtype=torch.float32)
    mu = torch.mean(X, dim=0)
    var = torch.var(X, dim=0, unbiased=False)

    X_new = (X - mu) / torch.sqrt(var + eps)

    Y = gamma * X_new +  beta

    return Y

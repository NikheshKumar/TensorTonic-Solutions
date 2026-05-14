import torch

def compute_loss(pred, target, method, delta=1.0):
    """
    Returns: float, the mean loss value
    """

    pred = torch.as_tensor(pred, dtype=torch.float32)
    target = torch.as_tensor(target, dtype=torch.float32)

    if method=="mse":
        l = torch.mean((pred-target)**2)
        
    if method=="cross_entropy":
        target = torch.as_tensor(target, dtype=torch.long)
        l = torch.nn.functional.cross_entropy(pred, target)
        l = torch.mean(l)
        
    if method=="huber":
        error = torch.abs(pred-target) 
        l = torch.where(error <=delta, 0.5 * (error ** 2), delta * (error - 0.5 * delta))
        l = torch.mean(l)


    return l.item()

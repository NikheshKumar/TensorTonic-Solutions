import torch
import torch.nn as nn

def manual_train_step(model, X, y, criterion, lr):
    """
    Returns: loss value as a Python float
    """
    out = model(X)
    loss = criterion(out, y)
    loss.backward()

    with torch.no_grad():
        for p in model.parameters():
            p -= lr*p.grad

    model.zero_grad()

    return float(loss.item())
    

    

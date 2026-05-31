import torch
import torch.nn as nn

def train_epoch(model, dataloader, criterion, optimizer):
    """
    Returns: average loss over all batches (float)
    """
    loss = 0.0
    n = 0
    model.train()

    for i, targets in dataloader:
        optimizer.zero_grad()
        out = model(i)
        l = criterion(out, targets)
        l.backward()
        optimizer.step()

        loss += l
        n += 1

    return float(loss / n)

import torch
import torch.nn as nn

def train_with_scheduler(model, dataloader, criterion, optimizer, scheduler, num_epochs):
    """
    Returns: dict with 'losses' (list of per-epoch avg loss) and 'lrs' (list of learning rate per epoch)
    """
    loss = []
    learning_rates = []

    for i in range(num_epochs):
        model.train()
        lr = optimizer.param_groups[0]["lr"]
        learning_rates.append(lr)

        tot = 0.0
        n = 0
        for i, targets in dataloader:
            optimizer.zero_grad()
            out = model(i)
            l = criterion(out, targets)
            l.backward()
            optimizer.step()
            tot += l
            n += 1

        loss.append(tot/n)
        scheduler.step()

    return {"losses":loss, "lrs":learning_rates}
            
            
        
    

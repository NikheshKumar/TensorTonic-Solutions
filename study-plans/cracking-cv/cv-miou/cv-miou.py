import torch

def mean_iou(pred: list, target: list, num_classes: int) -> float:
    """
    Returns the mean IoU over present classes as a float, rounded to 4 decimals.
    """
    pred = torch.tensor(pred, dtype=torch.long)
    target = torch.tensor(target, dtype=torch.long)

    res = []
    
    for c in range(num_classes):
        pred_c = (pred   == c)
        tgt_c  = (target == c)
        
        tp = (pred_c & tgt_c).sum().item()
        fp = (pred_c & ~tgt_c).sum().item()
        fn = (~pred_c & tgt_c).sum().item()

        den = tp + fp + fn
        if den==0.0:
            continue 
        res.append(tp/den)

    if not res:
        return 0.0

    return float(round(sum(res)/len(res), 4))
    
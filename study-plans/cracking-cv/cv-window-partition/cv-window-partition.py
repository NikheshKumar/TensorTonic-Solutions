import torch

def window_partition_reverse(x: list, window_size: int) -> dict:
    """
    Returns a dictionary with partitioned and reversed, both nested float lists.
    """
    x = torch.tensor(x, dtype=torch.float64)

    B, H, W, C = x.shape

    nh, nw = H//window_size, W//window_size

    new_x = x.reshape(B, nh, window_size, nw, window_size, C)
    new_x = new_x.permute(0, 1, 3, 2, 4, 5)
    new_x = new_x.reshape(B*nh*nw, window_size, window_size, C)
    partitioned = torch.round(new_x, decimals=4).tolist()
    
    new_x = new_x.reshape(B, nh, nw, window_size, window_size, C)
    new_x = new_x.permute(0, 1, 3, 2, 4, 5)
    reversed = torch.round(new_x.reshape(B, H, W, C), decimals=4).tolist()

    return {"partitioned":partitioned, "reversed":reversed}

    
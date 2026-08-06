import torch

def skipgram_pairs(token_ids: torch.Tensor, window: int) -> torch.Tensor:
    """
    Returns int64 torch.Tensor of shape (num_pairs, 2).
    """
    # YOUR CODE HERE
    ans = []
    
    for i in range(token_ids.shape[0]):
        a = max(0, i-window)
        b = min(token_ids.shape[0], i+window+1)
        for j in range(a,b):
            if i!=j:
                ans.append([token_ids[i].item(), token_ids[j].item()])

    if not ans:
        return torch.zeros((0, 2), dtype=torch.int64)


    return torch.tensor(ans, dtype=torch.int64)

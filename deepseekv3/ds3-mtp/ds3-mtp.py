import torch

def mtp_head(h: torch.Tensor, W_projs: list, W_head: torch.Tensor) -> torch.Tensor:
    """
    Returns: torch.Tensor of shape (batch, seq_len, num_predict, vocab_size)
    """
    # YOUR CODE HERE
    B, seq_len, D = h.shape
    num_predict = len(W_projs)
    vocab_size = W_head.shape[0]

    W_projs = torch.stack(W_projs, dim=0)

    h = h.unsqueeze(2)

    W_projs = W_projs.unsqueeze(0).unsqueeze(0)
    
    h_d = (h.unsqueeze(-2) @ W_projs.transpose(-1, -2)).squeeze(-2)

    logits = h_d @ W_head.T

    return logits
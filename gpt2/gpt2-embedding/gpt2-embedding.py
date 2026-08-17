import torch
import torch.nn as nn

def gpt2_embedding(token_ids, token_embed_weight, position_embed_weight):
    """
    Returns: torch.Tensor of shape (seq_len, d_model)
    """

    token_embed_weight = torch.tensor(token_embed_weight, dtype=torch.float32)
    token_ids = torch.tensor(token_ids, dtype=torch.long)
    position_embed_weight = torch.tensor(position_embed_weight, dtype=torch.float32)
    
    seq_len = token_ids.shape[0]
    
    output = token_embed_weight[token_ids] + position_embed_weight[:seq_len]

    return output

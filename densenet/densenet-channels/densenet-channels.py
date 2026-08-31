import math
import torch

def densenet_channel_counts(stem_channels: int, growth_rate: int, block_layers, compression: float) -> torch.Tensor:
    """
    Returns a 1D int64 torch.Tensor of channel counts at each stage.
    """
    # YOUR CODE HERE

    C_block = stem_channels
    block_layers = list(block_layers)
    channel_counts = [C_block]
    
    for i in range(len(block_layers)-1):
        C_block += block_layers[i] * growth_rate
        channel_counts.append(C_block)
        C_block = math.floor(compression * C_block)
        channel_counts.append(C_block)

    C_block += block_layers[-1] * growth_rate
    channel_counts.append(C_block)

    return torch.tensor(channel_counts, dtype=torch.long)
        
    

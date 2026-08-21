import torch
from typing import Tuple

def symmetric_int8_quantize(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns: (quantized int8 tensor, scale scalar tensor, dequantized float tensor)
    """
    if x.abs().max().item()==0.0:
        scale = torch.tensor(1, dtype=torch.float32)
        q = torch.tensor(x, dtype=torch.int8)
        
        return q, scale, x
        
    scale = torch.max(torch.abs(x)) / 127

    q = torch.clip(torch.round(x / scale), -127.0, 127.0)
    q = torch.tensor(q, dtype=torch.int8)

    x_dq = q * scale

    return q, scale, x_dq

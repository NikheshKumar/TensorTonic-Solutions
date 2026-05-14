import torch
import torch.nn as nn

class CustomLinear(nn.Module):
    """
    Returns: y = x W^T + b without using nn.Linear
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(6))
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(self.weight)
        fan_in = float(self.weight.size(1))
        bound = 1/torch.sqrt(torch.tensor(fan_in)) if fan_in > 0.0 else nn.init.zeros_(self.bias)
    

    def forward(self, x):
        
        x = x @ self.weight.T + self.bias

        return x
        

import torch
import torch.nn as nn

class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p 
        

    def forward(self, x):
        """
        Returns: tensor with dropout applied
        """
        if self.p==0.0 or not self.training:
            return x
            
        if self.p == 1.0:
            return torch.zeros_like(x)
        
        m = torch.rand(x.shape) >= self.p
        m = m.to(x.dtype)

        return x * m / (1-self.p)
        

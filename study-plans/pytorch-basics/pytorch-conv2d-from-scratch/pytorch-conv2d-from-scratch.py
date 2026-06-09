import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        """
        Returns: None
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        """
        Returns: convolved output tensor of shape (batch, out_channels, H-k+1, W-k+1)
        """
        res = F.conv2d(x, self.weight, bias=self.bias, stride=1, padding=0)

        return res
        

import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        """
        Returns: None
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, 
                               kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, 
                               kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        
        
    
    def forward(self, x):
        """
        Returns: output tensor
        """

        y = x.clone()
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + y 

        out = self.relu(out)

        return out

        
        
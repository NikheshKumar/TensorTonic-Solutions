import numpy as np

def relu(x):
    return np.maximum(0, x)

class BottleneckBlock:
    """
    Bottleneck Block: 1x1 -> 3x3 -> 1x1
    Reduces computation by compressing channels.
    """
    
    def __init__(self, in_channels: int, bottleneck_channels: int, out_channels: int):
        self.in_ch = in_channels
        self.bn_ch = bottleneck_channels  # Compressed dimension
        self.out_ch = out_channels
        
        # 1x1 reduce
        self.W1 = np.random.randn(in_channels, bottleneck_channels) * 0.01
        # 3x3 (simplified as dense)
        self.W2 = np.random.randn(bottleneck_channels, bottleneck_channels) * 0.01
        # 1x1 expand
        self.W3 = np.random.randn(bottleneck_channels, out_channels) * 0.01
        
        # Shortcut (if dimensions differ)
        self.Ws = np.random.randn(in_channels, out_channels) * 0.01 if in_channels != out_channels else None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Bottleneck forward: compress -> process -> expand + skip
        """
        # YOUR CODE HERE
        x = np.asarray(x, float)
        
        h = relu(x@self.W1)

        z1 = relu(h@self.W2)

        z2 = z1 @ self.W3

        s = x@self.Ws if self.in_ch!=self.out_ch else x

        y = relu(z2 + s)

        return y
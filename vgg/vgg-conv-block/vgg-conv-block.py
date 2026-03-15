import numpy as np

def vgg_conv_block(x: np.ndarray, num_convs: int, out_channels: int) -> np.ndarray:
    """
    Implement a VGG-style convolutional block.
    """
    # Your implementation here
    
    x = np.asarray(x, float)

    for _ in range(num_convs):

      B,H,W,C_in = x.shape
      C_out = out_channels
      x_pad = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)), mode='constant')

      #conv layers
      bias = np.zeros(C_out)
      weight = np.random.randn(3, 3, C_in, C_out) * 0.01
      y = np.zeros((B, H, W, C_out))

      for i in range(H):
        for j in range(W):
          patch = x_pad[:, i:i+3, j:j+3, :]
          
          y[:, i, j, :] = np.tensordot(patch, weight, axes=((1, 2, 3), (0, 1, 2))) + bias
      

      #relu layer
      x = np.maximum(0,y)

    return x
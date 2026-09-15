import numpy as np

def vgg_forward(x: np.ndarray, config: list, kernels: list,
                biases: list, classifier: dict) -> np.ndarray:
    """
    Returns float64 class logits with shape (B, C_classes).
    """
    output = x.copy()

    def conv(output, kernel, size, bias, pad):
        
        B, H, W, C = output.shape
        padded_image = np.pad(output, ((0,0),(pad,pad),(pad,pad),(0,0)))
        new_output = np.zeros( (B, H, W, kernel.shape[3]), dtype=np.float64)
        
        for i in range(H):
            for j in range(W):
                window = padded_image[:, i:i+size, j:j+size, :]
                new_output[:, i, j, :] = np.einsum('nchw,chwk->nk', window, kernel) + bias

        output = np.maximum(0.0, new_output)
        return output

    for i, l in enumerate(config):
        
        if l=="M":
            B, H, W, C = output.shape
            output = output.reshape(B, H//2, 2, W//2, 2, C)
            output = np.max(output, axis=(2,4))
            
        else:
            B, H, W, C = output.shape
            kernel = kernels[i]
            bias = biases[i]
            size = kernel.shape[0]
            pad = kernel.shape[0] // 2
            output = conv(output, kernel, size, bias, pad)


    output = output.reshape(B, -1)
    y = np.maximum(0, output @ classifier["W1"] + classifier["b1"])
    y = np.maximum(0, y @ classifier["W2"] + classifier["b2"])
    y = y @ classifier["W3"] + classifier["b3"]

    return y
    
    
        

        
import numpy as np

def resnet_forward(x, conv1, W1_b1, W2_b1, W1_b2, W2_b2, Ws_b2, fc):
    """
    Returns the network logits as a nested list.
    """
    x = np.asarray(x, dtype=np.float64)
    conv1 = np.asarray(conv1, dtype=np.float64)
    W1_b1 = np.asarray(W1_b1, dtype=np.float64)
    W2_b1 = np.asarray(W2_b1, dtype=np.float64)
    W1_b2 = np.asarray(W1_b2, dtype=np.float64)
    W2_b2 = np.asarray(W2_b2, dtype=np.float64)
    Ws_b2 = np.asarray(Ws_b2, dtype=np.float64)
    fc = np.asarray(fc, dtype=np.float64)

    def _relu(z):
        return np.maximum(0,z)
    
    # conv1 
    y = _relu(x @ conv1)

    # block1
    id1 = y                                      
    z   = _relu(y @ W1_b1)                
    z   = z @ W2_b1                              
    z1  = _relu( z + id1)  
    
    # block 2
    id2 = z1 @ Ws_b2
    z   = _relu(z1 @ W1_b2)              
    z   = z @ W2_b2                              
    z2  = _relu(z + id2 )  

    # final result
    res = z2 @ fc

    return np.round(res, 4).tolist()
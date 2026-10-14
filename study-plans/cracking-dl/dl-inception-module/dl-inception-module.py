import numpy as np

def inception_module(x: list, W_1x1: list, b_1x1: list, W_3x3: list, b_3x3: list) -> dict:
    """
    Returns a dictionary of nested lists: branch_1x1, branch_3x3, output.
    """
    x = np.array(x, dtype=np.float64)
    W_1x1 =  np.array(W_1x1, dtype=np.float64)
    b_1x1 =  np.array(b_1x1, dtype=np.float64)
    W_3x3 =  np.array(W_3x3, dtype=np.float64)
    b_3x3 =  np.array(b_3x3, dtype=np.float64)

    C_in, H, W = x.shape
    C_out1 = W_1x1.shape[0]
    C_out3 = W_3x3.shape[0]


    # branch 1
    branch1 = np.tensordot(W_1x1, x, axes=([1], [0])) 
    branch1 += b_1x1[:, None, None]
    branch1 = np.maximum(branch1, 0.0)
    
    # branch 2
    x_new = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    branch2 = np.zeros((C_out3, H, W), dtype=np.float64)
    
    for i in range(3):
        for j in range(3):
            window = x_new[:, i:i+H, j:j+W]                 
            branch2 += np.tensordot(W_3x3[:, :, i, j], window, axes=([1], [0])) 
            
    branch2 += b_3x3[:, None, None]
    branch2 = np.maximum(branch2, 0.0)
    
    output = np.concatenate([branch1, branch2], axis=0)

    return {"branch_1x1":np.round(branch1,4).tolist(), "branch_3x3":np.round(branch2,4).tolist(), "output":np.round(output,4).tolist()}

    

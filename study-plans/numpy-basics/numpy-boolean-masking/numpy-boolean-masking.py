import numpy as np

def row_summary(data, threshold):
    """Returns: np.ndarray of shape (3, m, n), stacked element mask, any-filtered, all-filtered"""
    
    new_data = np.array(data, dtype=np.float64, ndmin=2)
    m, n= new_data.shape
    

    #element level masking
    ele_mask = new_data > threshold

    #row level masking
    row_mask1 = np.any(ele_mask, axis=1, keepdims=True)

    row_mask2 = np.all(ele_mask, axis=1, keepdims=True)

    res = np.stack([ele_mask, row_mask1 * new_data, row_mask2 * new_data], axis=0)

    return res.astype(np.float64)
    

    

    
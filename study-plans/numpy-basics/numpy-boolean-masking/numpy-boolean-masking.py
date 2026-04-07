import numpy as np

def row_summary(data, threshold):
    """Returns: np.ndarray of shape (3, m, n), stacked element mask, any-filtered, all-filtered"""
    
    new_data = np.array(data, dtype=np.float64, ndmin=2)
    m, n= new_data.shape
    

    #element level masking
    ele_mask = new_data > threshold

    #row level masking
    row_mask1 = np.any(new_data > threshold, axis=1, keepdims=True)
    row_any = np.broadcast_to(row_mask1, (m, n))
    row_any_filtered = row_any * new_data

    row_mask2 = np.all(new_data > threshold, axis=1, keepdims=True)
    row_all = np.broadcast_to(row_mask2, (m, n))
    row_all_filtered = row_all * new_data

    res = np.stack([ele_mask, row_any_filtered, row_all_filtered], axis=0)

    return res.astype(np.float64)
    

    

    
def pad_and_truncate(sequences, max_length, pad_value=0):
    """
    Returns: list[list[int]]
    """
    import numpy as np 

    res = []
    
    for s in sequences:
        s = np.asarray(s)
        
        if len(s)>max_length:
            res.append(s[:max_length].tolist())
            
        else:
            padded_seq = np.full(max_length - len(s), pad_value)
            new_s = np.concatenate([s, padded_seq])
            res.append(new_s.tolist())

    return res
        
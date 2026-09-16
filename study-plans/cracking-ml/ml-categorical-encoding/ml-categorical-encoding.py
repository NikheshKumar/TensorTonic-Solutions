import numpy as np

def categorical_encode(data, method="label"):
    """
    Returns: encoded result based on method
    """
    if method=="label":
        
        unique = sorted(set(data))
        dic = {u:i for i, u in enumerate(unique)}
        encoded = [dic[c] for c in data]

        return {"encoded":encoded, "classes":unique}

    elif method=="onehot":

        cla, encoded = np.unique(data, return_inverse=True)
        n_cla = len(cla)
        out = np.zeros((len(data), n_cla), dtype=np.int8)
        out[np.arange(len(data)), encoded] = 1

        return out
            
            
            
        
import numpy as np

def weight_init_params(layer_dims, method):
    """
    Returns: list of dicts with keys 'fan_in', 'fan_out', 'shape', 'scale'
    """

    res = []

    for n_in, n_out in zip(layer_dims, layer_dims[1:]):
        
        fan_in = n_in
        fan_out = n_out
        shape = [fan_out, fan_in]

        if method == "random_normal":
                scale = 1.0
            
        if method == "xavier_normal":
                scale = np.sqrt(2.0/(n_in + n_out))
                
        if method == "xavier_uniform":
                scale = np.sqrt(6.0/(n_in + n_out))
            
        if method == "kaiming_normal":
                scale = np.sqrt(2.0/(n_in))
            
        if method == "kaiming_uniform":
                scale = np.sqrt(6.0/(n_in))

        res.append({ "fan_in":fan_in, "fan_out":fan_out, "shape":shape, "scale": np.round(scale, 4) } )
            


    return res
        
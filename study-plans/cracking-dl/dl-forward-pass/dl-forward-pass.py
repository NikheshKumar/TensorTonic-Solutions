import numpy as np

def forward_pass(x, weights, biases):
    """
    Returns: Dict with "activations" and "pre_activations", values rounded to 4 decimals.
    """

    a = x.copy()

    activations = [a]
    pre_activations = []


    for i in range(len(weights)):
        
        W = np.asarray(weights[i], dtype=np.float64)
        b = np.asarray(biases[i], dtype=np.float64)
        z = W @ a + b

        pre_activations.append(z)

        if i < len(weights) - 1:
            a = np.maximum(0, z)
        else:
            a = z
        
        activations.append(a)


    
    return {
        "activations": [np.round(arr, 4).tolist() for arr in activations],
        "pre_activations": [np.round(arr, 4).tolist() for arr in pre_activations]
    }
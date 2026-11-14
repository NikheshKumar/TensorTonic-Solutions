import numpy as np

def jvp_vjp_2layer(W1: list, W2: list, x: list, x_dot: list, v_bar: list) -> dict:
    """
    Returns network output, JVP, VJP, and multiply counts.
    """
    W1 = np.asarray(W1, dtype=np.float64)
    W2 = np.asarray(W2, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    x_dot = np.asarray(x_dot, dtype=np.float64)
    v_bar = np.asarray(v_bar, dtype=np.float64)

    h, d = W1.shape
    m = W2.shape[0]
    
    z1 = W1 @ x
    z1_dot = W1 @ x_dot
    a1 = np.maximum(0.0, z1)
    f = W2 @ a1
    relu_mask = (z1 > 0.0).astype(np.float64)
    a1_dot = z1_dot * relu_mask 
    f_dot = W2 @ a1_dot

    a1_bar = W2.T @ v_bar
    z1_bar = a1_bar * relu_mask
    x_bar = W1.T @ z1_bar

    jvp_flops = h * d + h + m * h               
    vjp_flops = m * h + h + h * d               
    full_jacobian_jvp_flops = d * jvp_flops     
    full_jacobian_vjp_flops = m * vjp_flops


    def convert_to_list(z):
        return np.asarray(z).tolist()

    return {"f":convert_to_list(f), "jvp_result":convert_to_list(f_dot), "vjp_result":convert_to_list(x_bar), "jvp_flops":jvp_flops, "vjp_flops":vjp_flops, "full_jacobian_jvp_flops":full_jacobian_jvp_flops, "full_jacobian_vjp_flops":full_jacobian_vjp_flops}

    

    
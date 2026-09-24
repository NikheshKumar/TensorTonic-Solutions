import numpy as np

def gan_step(real_data: list, z: list, W_g: list, b_g: list, W_d: list, b_d: list, lr: float) -> dict:
    """
    Returns updated W_g, b_g, W_d, b_d and scalar d_loss, g_loss.
    """
    real_data = np.array(real_data, dtype=np.float64)
    z = np.array(z, dtype=np.float64)
    W_g = np.array(W_g, dtype=np.float64)
    b_g = np.array(b_g, dtype=np.float64)
    W_d = np.array(W_d, dtype=np.float64)
    b_d = np.array(b_d, dtype=np.float64)

    B = real_data.shape[0]
    
    F = np.tanh(z @ W_g.T + b_g)

    a_real = real_data @ W_d.T + b_d

    a_fake = F @ W_d.T + b_d

    loss_d = np.mean(np.logaddexp(0.0, -a_real) + np.logaddexp(0.0, a_fake))

    d_real = -1.0 / (B * (1.0 + np.exp(a_real)))
    d_fake = 1.0 / (B * (1.0 + np.exp(-a_fake))) 

    dW_d = d_real.T @ real_data + d_fake.T @ F
    db_d = np.sum(d_real + d_fake, axis=0)

    W_d = W_d - lr * dW_d
    b_d = b_d - lr * db_d

    a_fake_new = F @ W_d.T + b_d

    loss_g = np.mean(np.logaddexp(0.0, -a_fake_new))

    dF = (-1.0 / (B * (1.0 + np.exp(a_fake_new)))) @ W_d

    dW_g = (dF * (1.0 - F**2)).T @ z
    db_g = (dF * (1.0 - F**2)).sum(axis=0)

    W_g = W_g - lr * dW_g
    b_g = b_g - lr * db_g

    
    return {"W_g": W_g.tolist(), "b_g": b_g.tolist(), "W_d": W_d.tolist(), "b_d": b_d.tolist(), "d_loss": float(loss_d), "g_loss": float(loss_g)}

    
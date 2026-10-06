import numpy as np

def get_2d_sincos_pos_embed(embed_dim: int, grid_h: int, grid_w: int) -> list:
    """
    Returns a float list of shape (grid_h * grid_w, embed_dim), rounded to 4 decimals.
    """
    d = embed_dim//2
    i = np.arange(0,d,2, dtype=np.float64)
    omega = 1e4 ** (-i / d)   

    def e(p):
        angles = p * omega
        z = np.concatenate([np.sin(angles), np.cos(angles)])
        return z

    out = []
    for i in range(grid_h):
        for j in range(grid_w):
            y = np.concatenate([e(i), e(j)])
            out.append(y)

    out = np.asarray(out, dtype=np.float64)

    return np.round(out,4).tolist()

    
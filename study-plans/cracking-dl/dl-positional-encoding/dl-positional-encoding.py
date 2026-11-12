import numpy as np

def rope(x: list, mode: str = 'forward', d_output: list | None = None) -> dict:
    """
    Returns forward results and optional backward gradients as a dictionary of lists.
    """
    x = np.asarray(x, dtype=np.float64)

    T, D = x.shape

    y = x.copy()

    t = np.arange(T)
    i = np.arange(0, D//2)

    base = 10000.0

    angles = t[:, None] * (base ** (-2.0 * i / D))[None, :]

    x_even = x[:, 0::2]
    x_odd = x[:, 1::2]

    cos = np.cos(angles)
    sin = np.sin(angles)

    y[:, 0::2] = x_even * cos - x_odd * sin

    y[:, 1::2] = x_even * sin + x_odd * cos

    def convert_and_round(z):
        return np.round(z,4).tolist()

    result = {"rotated":convert_and_round(y), "cos_pe":convert_and_round(cos), "sin_pe":convert_and_round(sin)}

    if mode=="backward":
        g = np.asarray(d_output, dtype=np.float64)
        g_odd = g[:, 1::2]
        g_even = g[:, 0::2]
        dx = g.copy()
        dx[:, 0::2] = g_even * cos + g_odd * sin
        dx[:, 1::2] = - g_even * sin + g_odd * cos
        result["dx"] = convert_and_round(dx)

    return result

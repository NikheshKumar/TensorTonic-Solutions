import numpy as np

def vit_encoder_block(x: np.ndarray, num_heads: int,
                      Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray,
                      Wo: np.ndarray, W1: np.ndarray, W2: np.ndarray) -> np.ndarray:
    """
    Returns the float64 output of one pre-normalized ViT encoder block.
    """
    def layernorm(x):
        eps = 1e-6
        m = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return (x - m) / np.sqrt(var + eps)

    def softmax(x, axis=-1):
        max_val = np.max(x, axis=axis, keepdims=True)
        num = np.exp(x - max_val)
        den = np.sum(num, axis=axis, keepdims=True)
        return num / den

    def gelu(x):
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


    B, seq_len, d_model = x.shape
    d_head = d_model//num_heads


    seq_len = x.shape[1]
    d_model = x.shape[2]
    d_head = d_model // num_heads

    x_norm1 = layernorm(x)

    Q = (x_norm1 @ Wq).reshape(B, seq_len, num_heads, d_head).transpose(0, 2, 1, 3)
    K = (x_norm1 @ Wk).reshape(B, seq_len, num_heads, d_head).transpose(0, 2, 1, 3)
    V = (x_norm1 @ Wv).reshape(B, seq_len, num_heads, d_head).transpose(0, 2, 1, 3)

    scores = Q @ np.transpose(K,(0, 1, 3, 2)) / (d_head**0.5)
    weights = softmax(scores, axis=-1)
    att = weights @ V 

    context = att.transpose(0, 2, 1, 3).reshape(B, seq_len, d_model)
    mha = context @ Wo

    x = x + mha

    x_norm2 = layernorm(x)
    mlp = gelu(x_norm2@W1)@W2

    x = x + mlp

    return x
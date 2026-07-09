import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    # Your code here
    mu = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    
    y = (x-mu)/np.sqrt(var + eps)

    return gamma*y + beta

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    # Your code here

    import math
    import numpy as np

    batch_size, seq_length, d_model = Q.shape
    d_k = d_model // num_heads

    Q_proj = (Q @ W_q).reshape(batch_size, seq_length, num_heads, d_k).transpose(0, 2, 1, 3)
    K_proj = (K @ W_k).reshape(batch_size, seq_length, num_heads, d_k).transpose(0, 2, 1, 3)
    V_proj = (V @ W_v).reshape(batch_size, seq_length, num_heads, d_k).transpose(0, 2, 1, 3)
    
    scores = Q_proj @ K_proj.transpose(0,1,3,2)
    scaled_scores = scores / math.sqrt(d_k)

    weights = softmax(scaled_scores, axis=-1)
    context = (weights @ V_proj).transpose(0, 2, 1, 3).reshape(batch_size, seq_length, d_model)

    mha = context @ W_o

    return mha
    

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    # Your code here
    relu = np.maximum(0, x@W1 + b1)

    return (relu @ W2 + b2)
    

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    # Your code here
    mha_out = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)
    
    x_new = layer_norm(x + mha_out, gamma1, beta1)
    
    ffn_out = feed_forward(x_new, W1, b1, W2, b2)

    output = layer_norm(x_new + ffn_out, gamma2, beta2)

    return output
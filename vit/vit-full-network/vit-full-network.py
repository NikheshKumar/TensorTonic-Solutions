import numpy as np

def vit_forward(image: np.ndarray, patch_size: int, num_heads: int,
                W_patch: np.ndarray, patch_bias: np.ndarray,
                cls_token: np.ndarray, pos_embed: np.ndarray,
                encoder_weights: list, W_head: np.ndarray) -> np.ndarray:
    """
    Returns float64 Vision Transformer logits with shape (B, C).
    """
    #patch embedding
    
    B, H, W, C = image.shape

    embed_dim = W_patch.shape[1]

    H_new = (H // patch_size) * patch_size
    W_new = (W // patch_size) * patch_size

    H, W = H_new, W_new
    
    image = image[:, :H, :W, :]
    
    N = H * W // patch_size**2

    image_patches = image.reshape(B, H//patch_size, patch_size, W//patch_size, patch_size, C)

    image_patches = image_patches.transpose(0, 1, 3, 2, 4, 5)

    image_patches = image_patches.reshape(B, N, patch_size * patch_size * C)

    z = image_patches @ W_patch + patch_bias 

    # position embedding

    B, N, D = z.shape
    
    cls_token = np.broadcast_to(cls_token,(B,1,D))

    z = np.concatenate([cls_token, z], axis=1)

    z = z + pos_embed

    #encoder block
    
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


    B, seq_len, d_model = z.shape
    d_head = d_model//num_heads

    for layer in encoder_weights:
        Wq = np.array(layer["Wq"], dtype=np.float64)
        Wk = np.array(layer["Wk"], dtype=np.float64)
        Wv = np.array(layer["Wv"], dtype=np.float64)
        Wo = np.array(layer["Wo"], dtype=np.float64)
        
        W1 = layer["W1"]
        W2 = layer["W2"]

        seq_len = z.shape[1]
        d_model = z.shape[2]
        d_head = d_model // num_heads

        x_norm1 = layernorm(z)

        Q = (x_norm1 @ Wq).reshape(B, seq_len, num_heads, d_head).transpose(0, 2, 1, 3)
        K = (x_norm1 @ Wk).reshape(B, seq_len, num_heads, d_head).transpose(0, 2, 1, 3)
        V = (x_norm1 @ Wv).reshape(B, seq_len, num_heads, d_head).transpose(0, 2, 1, 3)

        scores = Q @ np.transpose(K,(0, 1, 3, 2)) / (d_head**0.5)
        weights = softmax(scores, axis=-1)
        att = weights @ V 

        context = att.transpose(0, 2, 1, 3).reshape(B, seq_len, d_model)
        mha = context @ Wo

        z = z + mha

        x_norm2 = layernorm(z)
        mlp = (gelu(x_norm2 @ W1) @ W2) 

        z = z + mlp

    # classification head

    enc_new = z[:,0,:]
    
    m = np.mean(enc_new, axis=1, keepdims=True)
    
    var = np.mean((enc_new - m)**2, axis=1, keepdims=True)
    
    eps = 1e-6
    
    cls_tokens_norm = (enc_new - m)/np.sqrt(var+eps)

    logits = cls_tokens_norm @ W_head

    return logits

    

    

    
import numpy as np

def classification_head(encoder_output: np.ndarray,
                        W_head: np.ndarray) -> np.ndarray:
    """
    Returns float64 class logits with shape (B, C).
    """
    enc_new = encoder_output[:,0,:]

    m = np.mean(enc_new, axis=1, keepdims=True)
    var = np.mean((enc_new - m)**2, axis=1, keepdims=True)
    eps = 1e-6
    cls_tokens_norm = (enc_new - m)/np.sqrt(var+eps)

    logits = cls_tokens_norm @ W_head

    return logits
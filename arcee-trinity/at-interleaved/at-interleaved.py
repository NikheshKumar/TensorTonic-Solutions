def uses_rope(layer_idx: int, rope_ratio: int) -> bool:
    """
    Returns True if the layer should apply RoPE, False for NoPE.
    rope_ratio = 3 means 3 RoPE layers per 1 NoPE layer.
    """
    # YOUR CODE HERE
    if (layer_idx + 1) % (rope_ratio + 1):
        return True

    return False
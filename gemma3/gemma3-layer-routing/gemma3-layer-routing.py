def is_global_layer(layer_idx: int, local_ratio: int) -> bool:
    """
    Returns: True if global (full) attention, False if local (sliding window)
    """
    # YOUR CODE HERE
    if (layer_idx + 1) % (local_ratio + 1) ==0:
        return True
    return False
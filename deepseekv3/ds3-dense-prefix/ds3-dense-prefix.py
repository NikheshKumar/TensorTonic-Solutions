def is_dense_layer(layer_idx: int, num_dense_layers: int) -> bool:
    """
    Returns: True if dense FFN, False if MoE
    """
    # YOUR CODE HERE
    return True if layer_idx < num_dense_layers else False
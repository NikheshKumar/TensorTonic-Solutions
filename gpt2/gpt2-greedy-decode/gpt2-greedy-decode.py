import torch

def greedy_decode(input_ids, logits_map, num_steps):
    """
    Returns: list of int (full generated sequence including input_ids)
    """
    # YOUR CODE HERE
    input_ids = list(input_ids)
    
    for i in range(num_steps):
        keys = tuple(input_ids)
        if keys not in logits_map:
            break
        l = torch.tensor(logits_map[keys], dtype=torch.float64)
        next = torch.argmax(l).item()
        input_ids.append(next)

    return input_ids
import torch
from typing import Tuple, List, Dict
from collections import Counter

def bpe_train(text: str, target_vocab_size: int) -> Tuple[List[Tuple[int, int]], Dict[int, bytes]]:
    """Returns: Tuple of (merge_rules, vocab) where merge_rules is a list of (id_a, id_b) tuples and vocab maps token IDs to bytes"""

    vocab = {i:bytes([i]) for i in range(0,256)}


    tokens = list(text.encode("utf-8"))

    merge_rules = []

    while len(vocab) < target_vocab_size:

        if len(tokens)<2:
            break
        
        counts = Counter((tokens[i], tokens[i+1]) for i in range(len(tokens)-1))

        if not counts:
            break

        a,b = min(counts.keys(), key=lambda p: (-counts[p], p))

        new_id = len(vocab)
        vocab[new_id] = vocab[a] + vocab[b]

        merge_rules.append((a,b))

        new_tokens = []

        i = 0
        while i < len(tokens):

            if i < len(tokens) - 1 and tokens[i] == a and tokens[i+1] == b:
                new_tokens.append(new_id)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1

        tokens = new_tokens
            
    return (merge_rules, vocab)
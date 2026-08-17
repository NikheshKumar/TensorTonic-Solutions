import torch
from typing import Tuple, List, Dict

def bpe_encode(text: str, merge_rules: List[Tuple[int, int]], vocab: Dict[int, bytes]) -> List[int]:
    """
    Returns: List of integer token IDs
    """
    tokens = list(text.encode("utf-8"))

    for (a,b) in merge_rules:
        merged = vocab[a] + vocab[b]
        new_id = None
        for id, token_bytes in vocab.items():
            if token_bytes == merged:
                new_id = id

        new_tokens = []

        i = 0
        while i< len(tokens):
            if i<len(tokens)-1 and tokens[i] == a and tokens[i+1]==b:
                new_tokens.append(new_id)
                i += 2

            else:
                new_tokens.append(tokens[i])
                i+=1

        tokens = new_tokens
                
    return tokens
    

def bpe_decode(token_ids: List[int], vocab: Dict[int, bytes]) -> str:
    """
    Returns: Decoded UTF-8 string
    """

    chunks = [vocab[token_id] for token_id in token_ids]

    text = b"".join(chunks).decode("utf-8")

    return text

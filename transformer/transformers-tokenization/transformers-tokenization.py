import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # YOUR CODE HERE

        sp_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        for i, t in enumerate(sp_tokens):
            self.word_to_id[t] = i
            self.id_to_word[i] = t

        unique = set()
        for t in texts:
            w = t.lower().split()
            for token in w:
                if token not in self.word_to_id:
                    unique.add(token)

        for w in sorted(unique):
            i = len(self.word_to_id)
            self.word_to_id[w] = i
            self.id_to_word[i] = w

        self.vocab_size = len(self.word_to_id)
        
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        # YOUR CODE HERE
        words = text.lower().split()

        enc = []
        unk_id = self.word_to_id[self.unk_token]
        
        for w in words:
            enc_id = self.word_to_id.get(w, unk_id)
            enc.append(enc_id)

        return enc
            
            
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        # YOUR CODE HERE
        res = []
        
        for i in ids:
            word = self.id_to_word.get(i, self.unk_token)
            res.append(word)

        return " ".join(res)
            

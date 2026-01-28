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
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0

        special_tokens = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token,
        ]

        for token in special_tokens:
            self.word_to_id[token] = self.vocab_size
            self.id_to_word[self.vocab_size] = token
            self.vocab_size += 1


        for t in texts:
            for word in t.split():
                if word not in self.word_to_id:
                    self.word_to_id[word] = self.vocab_size 
                    self.id_to_word[self.vocab_size] = word 
                    self.vocab_size += 1


    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        # YOUR CODE HERE

        tokens = text.split()
        unk_id = self.word_to_id[self.unk_token]

        enc = [self.word_to_id.get(word, unk_id) for word in tokens]

        return enc

    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        # YOUR CODE HERE

        dec = [self.id_to_word.get(i, self.unk_token) for i in ids]

        return " ".join(dec)
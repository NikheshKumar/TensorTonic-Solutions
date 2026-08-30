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

        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]

        for i, tok in enumerate(special_tokens):
            self.word_to_id[tok] = i
            self.id_to_word[i] = tok

        unique_words = set()

        for t in texts:
            w = t.lower().split()
            for token in w:
                if token not in self.word_to_id:
                    unique_words.add(token)
                

        for word in sorted(unique_words):
            i = len(self.word_to_id)
            self.word_to_id[word] = i
            self.id_to_word[i] = word

        self.vocab_size = len(self.word_to_id)
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        # YOUR CODE HERE
        text_new = text.lower().split()

        enc = []

        unk_id = self.word_to_id[self.unk_token]

        for w in text_new:
            enc_id = self.word_to_id.get(w, unk_id)
            enc.append(enc_id)

        return enc
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        # YOUR CODE HERE
        dec = []

        for i in ids:
            word = self.id_to_word.get(i, self.unk_token)
            dec.append(word)

        return " ".join(dec)

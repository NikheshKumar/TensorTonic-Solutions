from typing import List, Dict

class WordPieceTokenizer:
    """
    WordPiece tokenizer for BERT.
    """
    
    def __init__(self, vocab: Dict[str, int], unk_token: str = "[UNK]", max_word_len: int = 100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_word_len = max_word_len
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into WordPiece tokens.
        """
        tokens = []
        for word in text.lower().split():
            word_tokens = self._tokenize_word(word)
            tokens.extend(word_tokens)
        return tokens
    
    def _tokenize_word(self, word: str) -> List[str]:
        """
        Tokenize a single word into subwords.
        """
        # YOUR CODE HERE

        if len(word) > self.max_word_len:
            return [self.unk_token]

        output = []
        
        start = 0

        while start < len(word):
            end = len(word)
            curr = None

            while start < end:
                sub = word[start:end]
                
                if start>0:
                    sub = "##" + sub

                if sub in self.vocab:
                    output.append(sub)
                    curr = sub 
                    break 
                    
                end -= 1
    
            if curr is None:
                return [self.unk_token]

            start = end

        return output

        
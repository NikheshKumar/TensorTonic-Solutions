def bigram_probabilities(tokens):
    """
    Returns: (counts, probs)
      counts: dict mapping (w1, w2) -> integer count
      probs: dict mapping (w1, w2) -> float P(w2 | w1) with add-1 smoothing
    """
    # Your code here
    import numpy as np 
    from collections import Counter

    vocab = set(tokens)
    
    bigrams = []
    for i in range(len(tokens) - 1):
        bigrams.append((tokens[i], tokens[i+1]))
    
    counts = dict(Counter(bigrams))

    starting_word_count = Counter(tokens[:-1])
    
    probs = {}

    for w1 in vocab:
        for w2 in vocab:
            num = counts.get((w1, w2), 0) + 1
            den = starting_word_count[w1] + len(vocab)
            probs[(w1, w2)] = num / den

    
    return counts, probs
        
        
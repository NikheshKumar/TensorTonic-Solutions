def subword_tokenize(word: str, vocab: list) -> list:
    """
    Returns a list of subword strings, including continuation markers.
    """

    subwords = []
    v = set(vocab)
    
    n = len(word)
    i = 0
    while i < n:
        
        longest = None
        for j in range(n, i, -1):
            candidate = word[i:j] if i==0 else "##" + word[i:j]
            if candidate in v:
                longest = j
                break
                
        if longest is None:
            subwords.append(word[i] if i==0 else "##" + word[i])
            i += 1
            continue 
            
        subwords.append(word[i:longest] if i==0 else "##" + word[i:longest])
        i = longest


    return subwords
            

        
            
        
    
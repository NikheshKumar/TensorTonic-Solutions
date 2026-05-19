def tokenize(text):
    """
    Returns: list[str]
    """
    import numpy as np 


    if text is None:
        return []
        
    if hasattr(text, 'item'): 
        text = str(text.item())
    else:
        text = str(text)


    n = len(text)
    
    if n == 0:
        return []
    i = 0
    tokens = []

    while i < n:
        
        if text[i].isspace():
            i += 1
        
        elif (text[i].isalnum() or text[i]=='_'):
            w = ""
            while i < n and (text[i].isalnum() or text[i]=="_"):
                w += text[i]
                i += 1
            tokens.append(w)
            
        else:
            tokens.append(text[i])
            i += 1

    return tokens
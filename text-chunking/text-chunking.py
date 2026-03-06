def text_chunking(tokens, chunk_size, overlap):
    """
    Split tokens into fixed-size chunks with optional overlap.
    """
    # Write code here

    import numpy as np 

    step = chunk_size - overlap
    ans = []
    i = 0
  
    while i<len(tokens):
        
      tok = tokens[i:i+chunk_size]
      ans.append(tok)
      
      if i + chunk_size >= len(tokens):
        break
         
      i += step


    return ans
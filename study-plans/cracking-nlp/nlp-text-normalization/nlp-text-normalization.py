import re

def text_normalize(text, operations):
    """
    Returns: str
    """
    if text is None:
        return []
        
    if hasattr(text, 'item'): 
        text = str(text.item())
    else:
        text = str(text)


    for op in operations:
        if op == 'lowercase':
            text = text.lower()
        if op == 'remove_punctuation':
            text = re.sub(r'[^\w\s]', '', text)
        if op == 'remove_digits':
            text = re.sub(r'\d', '', text)
        if op == 'collapse_whitespace':
            text = re.sub(r'\s+', ' ', text)
        if op == 'strip' :
            text = text.strip()
            

    return text
        
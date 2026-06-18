import re

def extract_patterns(text, pattern_type):
    """
    Returns: list[str]
    """
    if pattern_type is None:
        return []
    if pattern_type == "emails" :
        return re.findall(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b', text)
    if pattern_type == "urls":
        return re.findall(r'https?://[^\s,)]+', text)
    if pattern_type == "hashtags":
        return re.findall(r'\#\w+', text)
    if pattern_type == "money":
        return re.findall(r'\$\d+(?:\.\d{2})?', text)
    if pattern_type == "dates":
        return re.findall(r'\b(?:\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{1,2}/\d{1,2}/\d{2,4})\b', text)
        
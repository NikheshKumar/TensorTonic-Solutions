def word_count_dict(sentences):
    """
    Returns: dict[str, int] - global word frequency across all sentences
    """
    # Your code here

    freq_dic = {}

    for s in sentences:
        for w in s:
            if w not in freq_dic:
                freq_dic[w] = 0 
            freq_dic[w] += 1

    return freq_dic
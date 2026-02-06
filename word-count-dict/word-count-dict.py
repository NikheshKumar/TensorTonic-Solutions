def word_count_dict(sentences):
    """
    Returns: dict[str, int] - global word frequency across all sentences
    """
    # Your code here
    import numpy as np 

    freq_dic = {}

    for sentence in sentences:
        for word in sentence:
            if word in freq_dic:
                freq_dic[word] += 1
            else:
                freq_dic[word] = 1


    return freq_dic            
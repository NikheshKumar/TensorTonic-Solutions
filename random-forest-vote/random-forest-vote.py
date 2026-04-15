import numpy as np

def random_forest_vote(predictions):
    """
    Compute the majority vote from multiple tree predictions.
    """
    # Write code here
    predictions = np.asarray(predictions, dtype=int)

    num_trees, num_samples = predictions.shape

    majority_votes = []

    for i in range(num_samples):
        sample_cols = predictions[:, i]
        
        counts = {}
        for val in sample_cols:
            counts[val] = counts.get(val, 0) + 1
        
        max_count = max(counts.values())
        
        candidates = [label for label, count in counts.items() if count == max_count]
        
        winner = min(candidates)
        majority_votes.append(winner)
        
    return majority_votes
            

    
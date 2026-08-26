import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

def create_balanced_loader(features, labels, batch_size):
    """
    Returns: a DataLoader that oversamples underrepresented classes
    """
    
    class_counts = torch.bincount(labels)
    sample_weights = 1.0 / class_counts[labels]

    dataset = TensorDataset(features, labels)
    
    sampler = WeightedRandomSampler(sample_weights, num_samples=labels.shape[0])

    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    return loader
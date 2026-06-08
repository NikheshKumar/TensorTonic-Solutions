## <span style="font-size: 20px;">The Data Loading Architecture in PyTorch</span>

- Training a neural network requires feeding data to the model in an organized, efficient, and repeatable manner
- PyTorch addresses this through a layered architecture of three core abstractions: Dataset, Sampler, and DataLoader
- **Dataset:** knows how to read a single sample from its underlying storage. Does not concern itself with batching, shuffling, or parallelism
- **Sampler:** decides the order in which samples are accessed by producing a sequence of indices that the DataLoader uses to request samples
- **DataLoader:** orchestrates everything - consults the Sampler for indices, fetches samples from the Dataset (optionally in parallel across worker processes), collates samples into batches, and yields batches to the training loop
- This separation of concerns means you can swap out any one component without touching the others. Replace the Dataset to read from a different format, change the Sampler to implement curriculum learning, or adjust worker count to tune throughput, all independently

## The Dataset Protocol

PyTorch defines two styles of datasets, but the most common is the map-style dataset. A map-style dataset implements two methods that form a protocol:

- The length method returns the total number of samples in the dataset. The DataLoader calls this to determine the number of batches in an epoch and to validate indices. It should return a plain integer and be fast, since it may be called many times
- The item-access method accepts an integer index and returns the corresponding sample. The return value is typically a tuple of tensors (features, label), but it can be any Python object. The DataLoader calls this once per sample per batch
- The contract: for any integer $i$ in the range $[0, \text{len}(dataset))$, accessing element $i$ should return the $i$-th sample
- The Dataset is a mapping from integers to samples. The DataLoader never accesses the dataset in any other way. This means your Dataset can read from CSV, SQL, image directories, HDF5, or any other source - as long as you can map an integer to a sample, the DataLoader does not care
- Although PyTorch provides a base class, it is not strictly required. The DataLoader checks for the existence of the item-access method using duck typing. The base class is useful for type hints and signaling intent, but the real requirement is the protocol

## Map-Style vs. Iterable-Style Datasets

- **Map-style** datasets support random access by index. They are appropriate when the full dataset resides on disk in a format that supports seeking, or when the dataset fits in memory. Most tabular, image classification, and structured datasets fall into this category
- **Iterable-style** datasets implement an iterator protocol instead of item access. They produce samples one at a time in a streaming fashion and do not require a length method. Designed for data sources where random access is impossible or impractical: network streams, log files being tailed in real time, or datasets too large to index
- The tradeoff is control vs. flexibility:
  - Map-style gives the Sampler full control over which samples appear in which order. Shuffling is trivial: generate a random permutation of indices
  - Iterable-style must handle shuffling internally (e.g., maintaining a shuffle buffer), and the Sampler is not used at all
- For the vast majority of training scenarios involving finite, indexable data, map-style datasets are the right choice

## Designing the Constructor: Eager vs. Lazy Loading

When you implement a custom Dataset, one of the first design decisions is what happens in the constructor. There are two broad strategies, and the right choice depends on the size of your data and the cost of reading individual samples.

## Eager Loading: Pre-process Everything Upfront

- The constructor reads all raw data, processes it, and stores results as tensors in memory
- When item access is called later, it simply indexes into pre-computed tensors
- For a tabular CSV dataset with $N$ rows and $D+1$ columns, convert the raw data to a float32 tensor of shape $(N, D+1)$ once during construction
- Tensor indexing with an integer index is an $O(1)$ operation that returns a view or small contiguous copy, both extremely fast
- Disadvantage: entire dataset must fit in RAM as tensors
- For tabular data this is rarely a problem: a dataset with $10^6$ rows and 100 float32 columns consumes roughly $10^6 \times 100 \times 4$ bytes $= 400$ MB

## Lazy Loading: Process On Demand

- The constructor stores only metadata or references (such as file paths)
- Item access performs the actual reading and processing each time it is called
- Standard approach for image datasets, where loading all images into memory would be prohibitive
- Advantage: minimal memory usage (only one sample at a time in memory)
- Disadvantage: every call incurs I/O and compute cost. This cost is typically hidden by DataLoader workers that prefetch samples in parallel

## Memory-Mapped Files

- For datasets too large for RAM but stored in a format that supports random access
- Libraries like NumPy memmap, HDF5, and Apache Arrow create a virtual array backed by a file on disk
- The operating system pages in data on demand; frequently accessed regions stay in the page cache
- From the Dataset's perspective, the code looks like eager loading (index into an array), but memory usage is governed by the OS page cache rather than explicit allocation
- Particularly valuable for NLP tasks where tokenized text corpora can be tens of gigabytes

## Tensor Data Types and Memory Considerations

- **float32:** default and most common dtype for features. Sufficient precision for the vast majority of neural network computations. Expected by most layers. Each element occupies 4 bytes
- **float64 (double precision):** rarely needed in deep learning. Doubles memory usage, and most GPU operations run significantly slower. Use only for specific scientific computing tasks requiring extra precision
- **float16 / bfloat16:** 2 bytes per element, used in mixed-precision training. Datasets are typically stored in float32 and cast to lower precision during the forward pass, not in the Dataset itself
- **long (int64):** standard dtype for classification labels, because cross-entropy loss and negative log likelihood loss expect integer targets. For regression tasks, labels should be float32
- Common mistake: storing classification labels as float32 and encountering cryptic errors from the loss function. Another mistake: using float64 for features when float32 suffices, wasting memory and causing dtype mismatch with model parameters

## Separating Features from Labels

In tabular datasets, the raw data is often a matrix where one column is the target variable and the rest are input features. Given a data matrix with $N$ rows and $D+1$ columns, where column $k$ is the label:

$$
\text{features}_i = [x_{i,0}, x_{i,1}, \ldots, x_{i,k-1}, x_{i,k+1}, \ldots, x_{i,D}]
$$

$$
\text{label}_i = x_{i,k}
$$

- Features tensor has shape $(N, D)$ and labels tensor has shape $(N, 1)$ or $(N,)$
- The choice between these two label shapes matters: if labels have shape $(N, 1)$, each individual label has shape $(1,)$. If labels have shape $(N,)$, each label is a scalar tensor with shape $()$
- When the DataLoader collates a batch, scalars stack into a 1D tensor of shape $(B,)$, while shape-$(1,)$ tensors stack into $(B, 1)$. Both are valid, but be consistent with what your loss function expects
- A general-purpose tabular dataset should accept the label column index as a constructor parameter rather than hardcoding it

## Handling Different Data Formats

The Dataset abstraction is agnostic to the underlying data format. The pattern adapts to common scenarios:

- **CSV and tabular data:** read the file in the constructor, parse into a list or NumPy array, convert to tensors, and store them. This is eager loading and works well for datasets up to a few gigabytes
- **Image directories:** store a list of (image_path, label) pairs in the constructor. In item access, load the image using PIL or OpenCV, apply transforms, and return the tensor and label. Torchvision provides ImageFolder for the standard directory structure where each class has its own subdirectory
- **Text data:** store tokenized sequences and labels. If pre-tokenized, this resembles tabular data. If tokenizing on the fly, item access calls the tokenizer. Variable-length sequences require a custom collate function for padding, or pre-padding to a fixed length
- **Multi-modal data:** a dataset combining images, text, and tabular features for a single sample. Item access returns a dictionary or named tuple with all modalities. The collate function must handle each modality appropriately

## The DataLoader: Batching, Shuffling, and Parallelism

The DataLoader bridges the Dataset and the training loop. Key parameters:

- **batch_size:** controls how many samples are collated into each batch. The DataLoader requests this many samples from the Dataset via the Sampler, then combines them using a collate function. For tensor data, the default collate stacks individual tensors along a new leading dimension, so $B$ tensors of shape $(D,)$ become one tensor of shape $(B, D)$
- **shuffle:** when True, the DataLoader uses a RandomSampler internally; when False, it uses a SequentialSampler. You cannot use shuffle=True simultaneously with a custom sampler, since the sampler controls ordering
- **drop_last:** when True, the final incomplete batch is discarded. Sometimes desirable for training (uniform batch sizes for batch normalization) but usually not for evaluation (want to score every sample)

## The Collate Function

- Receives a list of samples (each returned by item access) and produces a batch
- Default collate: recursively stacks tensors, concatenates numpy arrays, groups other types into lists. For a Dataset returning (features, label) tuples, produces (batched_features, batched_labels)
- Custom collate needed when: variable-length sequences must be padded, attention masks must be created, non-tensor data must be handled, or images of different sizes must be processed
- Passed via the collate_fn parameter of DataLoader

## Multi-Process Data Loading

- Default: DataLoader loads data in the main process (num_workers=0). Data loading and model computation happen sequentially: the GPU sits idle during data preparation
- For simple datasets like pre-loaded tabular tensors, this is fine because tensor indexing is nearly instantaneous
- For datasets reading images from disk, decoding them, and applying transforms, data loading time can dominate the training loop
- Setting num_workers > 0 spawns separate processes that prefetch batches in the background. While the GPU trains on one batch, workers prepare the next ones
- The DataLoader maintains a queue of prefetched batches (controlled by prefetch_factor, default 2 per worker)

## How Worker Processes Access the Dataset

- Each worker receives a copy of the Dataset object (via fork or spawn)
- If your Dataset holds data in memory (eager loading), each worker gets its own copy: with $W$ workers, you use roughly $W + 1$ times the memory. Solution: use memory-mapped storage or defer loading to item access
- If your Dataset opens file handles in the constructor, those handles may not be valid in forked child processes. The worker_init_fn parameter lets you run a function in each worker after it starts: re-open file handles, reseed RNGs, or partition data for iterable datasets
- For reproducibility with random augmentations, use a worker_init_fn to set seeds based on the worker ID, combined with a Generator passed to the DataLoader

## The Sampler Hierarchy

Samplers are iterators that yield indices. The DataLoader consumes indices from the Sampler and passes them to item access:

- **SequentialSampler:** yields indices $0, 1, 2, \ldots, N-1$. Default when shuffle=False
- **RandomSampler:** yields a random permutation. Default when shuffle=True. Supports replacement and configurable num_samples for truncating large datasets during development
- **SubsetRandomSampler:** takes a list of indices and yields a random permutation of that subset. Useful for train/validation splitting: generate a random split and give each subset to a different DataLoader
- **WeightedRandomSampler:** draws indices according to per-sample weight distribution. Primary tool for handling class imbalance
- **BatchSampler:** wraps another sampler and groups indices into batches. DataLoader uses this internally; rarely interact with it directly unless implementing custom batching
- **DistributedSampler:** partitions the dataset across processes for multi-GPU training so each GPU sees a disjoint subset

## Pin Memory and GPU Transfer

- Normally tensors reside in pageable CPU memory. When transferring to GPU, the CUDA runtime first copies data to a temporary pinned buffer, then initiates DMA transfer to GPU memory - a two-step process with overhead
- With pin_memory=True, the DataLoader allocates batch tensors directly in pinned memory, eliminating the first copy and allowing DMA to proceed immediately
- Nearly always beneficial when training on GPU; memory overhead is modest because only one or two batches are pinned at a time
- Pinned memory is a limited resource (cannot be swapped to disk), so using too much can reduce system performance

## Data Augmentation in the Dataset

- Augmentation (random transformations to training samples) is typically implemented inside the item-access method
- Because item access is called fresh for every sample in every epoch, each sample receives a different random transformation each time
- For image datasets: usually expressed as a transforms pipeline passed to the constructor and applied during item access
- Important: augmentation is applied only to training data. Pass different transform pipelines to training and validation Dataset instances: training includes random augmentations, validation includes only deterministic preprocessing

## Common Pitfalls and Best Practices

- Returning non-tensor data when the default collate function expects tensors will raise errors. Ensure every element of the returned tuple is a tensor or a type the collate function can handle (Python int/float are auto-converted)
- Mismatched dtypes between dataset and loss function: cross-entropy expects integer labels of type long, while MSE loss expects float32
- Forgetting to handle the last incomplete batch can cause shape mismatches in code that assumes fixed batch size
- Performing expensive operations in item access without using DataLoader workers means the GPU is starved for data
- Not applying transforms consistently between training and evaluation leads to silent accuracy degradation
- When using multiple workers with random augmentation, failing to seed workers properly means results are not reproducible

## Batching Strategies

The simplest batching strategy is uniform: take the next $B$ samples from the Sampler and stack them. More sophisticated strategies exist:

- **Length-based batching:** groups sequences of similar length into the same batch, minimizing padding. Particularly important in NLP, where padding short sequences to the longest in the batch wastes computation. A bucket batch sampler sorts samples by length, partitions into buckets, and batches within each bucket
- **Dynamic batching:** adjusts batch size based on total tokens or pixels rather than number of samples. This keeps GPU memory usage roughly constant regardless of sequence length or image size
- **Gradient accumulation:** when the desired effective batch size exceeds GPU memory, process multiple small batches, accumulate gradients, then perform a single optimization step. The DataLoader does not need to know about this; it yields small batches and the training loop handles accumulation

## Performance Tuning

The goal of performance tuning is to ensure that data loading never becomes the bottleneck. The GPU should always have a batch ready to process. Several levers are available:

- Increase num_workers until GPU utilization stops improving. A common starting point is the number of CPU cores, but the optimal value depends on the cost of item access and storage speed. SSD storage can typically saturate more workers than spinning disks
- Enable pin_memory=True when training on GPU. The overhead is negligible and the speedup from faster DMA transfers is consistent
- Increase prefetch_factor (default 2) if workers are fast but there are occasional stalls. A higher prefetch factor means more batches are prepared ahead of time
- Pre-process data into efficient formats: single LMDB, TFRecord, or WebDataset archives reduce overhead of opening many small files. Pre-tokenized memory-mapped arrays eliminate tokenization cost during training
- Profile the data pipeline to identify where time is spent. If 80% of training time is in data loading, no amount of model optimization will help

## Summary

The Dataset provides random access to individual samples via its length and item-access methods. The Sampler determines the order of access by yielding indices. The DataLoader brings it all together: it pulls indices from the Sampler, fetches samples from the Dataset (possibly in parallel across workers), collates them into batches, optionally pins memory, and yields batches to the training loop. For tabular data that fits in memory, eager loading in the constructor and simple tensor indexing in item access is both the simplest and the fastest approach. For larger or more complex data, lazy loading combined with multiple workers keeps the pipeline efficient. Understanding this architecture lets you write Datasets that are correct, efficient, and composable with the rest of the PyTorch ecosystem.

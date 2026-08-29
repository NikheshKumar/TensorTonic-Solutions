## The Contrastive Learning Framework

InfoNCE (Information Noise-Contrastive Estimation) is the loss function behind many modern self-supervised learning methods like SimCLR, MoCo, and CLIP. It learns representations by contrasting positive pairs against negative pairs.

The setup:
- **Anchor**: a sample we want to learn representations for
- **Positive**: a sample similar to the anchor (e.g., another augmentation of the same image)
- **Negatives**: samples dissimilar to the anchor (e.g., other images in the batch)

The goal: make the anchor close to its positive and far from all negatives.

---

## The InfoNCE Formula

For an anchor with embedding $z$, positive with embedding $z^+$, and $N$ negatives with embeddings $z^-_1, ..., z^-_N$:

$$
L = -\log \frac{\exp(\text{sim}(z, z^+) / \tau)}{\exp(\text{sim}(z, z^+) / \tau) + \sum_{i=1}^{N} \exp(\text{sim}(z, z^-_i) / \tau)}
$$

Where:
- sim(a, b) is a similarity function (usually cosine similarity or dot product)
- $\tau$ is the temperature parameter
- The denominator sums over the positive and all negatives

---

## Breaking Down the Formula

The structure is similar to softmax cross-entropy:

**Numerator:** similarity between anchor and positive, exponentiated
**Denominator:** sum of similarities between anchor and ALL samples (positive + negatives), exponentiated

This can be rewritten as:

$$
L = -\text{sim}(z, z^+)/\tau + \log\left(\sum_{k} \exp(\text{sim}(z, z_k)/\tau)\right)
$$

Where $k$ ranges over positive and all negatives.

The loss is minimized when:
- sim(z, z+) is large (anchor close to positive)
- sim(z, z-) is small for all negatives (anchor far from negatives)

---

## The Temperature Parameter

The temperature $\tau$ controls the "hardness" of the contrastive task:

**Low temperature (e.g., 0.07):**
- Sharper distribution
- Hard negatives dominate the gradient
- Model must distinguish very carefully
- Can be unstable if too low

**High temperature (e.g., 1.0):**
- Softer distribution
- All negatives contribute more equally
- Easier optimization but less discriminative

**Common values:** 0.07 to 0.5

The optimal temperature depends on:
- Number of negatives
- Difficulty of the task
- Embedding dimensionality

---

## Why It Works: Information Theory Perspective

InfoNCE has a deep connection to mutual information. The loss is a lower bound on:

$$
I(X; Y) \geq \log(N) - L_{\text{InfoNCE}}
$$

Where:
- $I(X; Y)$ is the mutual information between the anchor and positive
- $N$ is the number of negatives + 1

Minimizing InfoNCE maximizes a lower bound on mutual information. This means the learned representations capture information shared between different views of the same data.

---

## Numerical Example

Consider an anchor with 1 positive and 3 negatives:

Similarities (before temperature scaling):
- Anchor to positive: 0.9
- Anchor to negative 1: 0.3
- Anchor to negative 2: 0.1
- Anchor to negative 3: -0.2

With temperature = 0.5:

Scaled similarities: 1.8, 0.6, 0.2, -0.4

Exponentials: exp(1.8) = 6.05, exp(0.6) = 1.82, exp(0.2) = 1.22, exp(-0.4) = 0.67

Sum of denominator: 6.05 + 1.82 + 1.22 + 0.67 = 9.76

Loss = -log(6.05 / 9.76) = -log(0.62) = 0.48

If the positive were more similar (say 0.99 instead of 0.9), the loss would be lower.

---

## The Importance of Negatives

InfoNCE requires many negatives to work well:

**Few negatives (e.g., 10):**
- The task is too easy
- Random chance of picking the positive is 10%
- Model can succeed without learning good representations

**Many negatives (e.g., 65,536 in MoCo):**
- The task is harder
- Random chance of picking the positive is 0.0015%
- Model must learn meaningful features to distinguish

**The batch size dilemma:**
- More negatives = better learning
- But larger batches are expensive
- Solutions: memory banks (MoCo), large batch training (SimCLR)

---

## Symmetric InfoNCE

In many implementations (like SimCLR), the loss is computed symmetrically:

For a pair (i, j) of augmented views:
- Compute loss with i as anchor, j as positive
- Compute loss with j as anchor, i as positive
- Average the two losses

This ensures both views learn equally good representations.

---

## The Gradient

The gradient with respect to the anchor embedding $z$:

$$
\frac{\partial L}{\partial z} = -\frac{1}{\tau}\left(z^+ - \sum_k p_k \cdot z_k\right)
$$

Where $p_k$ is the softmax probability for sample $k$.

Interpretation:
- Pull anchor toward the positive
- Push anchor away from the weighted average of all samples
- Samples with higher similarity (hard negatives) get pushed harder

---

## InfoNCE vs. Triplet Loss

**Triplet loss:**
- Uses one positive and one negative per anchor
- Margin-based: only cares about relative ordering
- Requires careful mining of hard triplets

**InfoNCE:**
- Uses one positive and many negatives
- Softmax-based: considers all negatives simultaneously
- More stable gradients, less need for mining
- Scales better with batch size

InfoNCE generally outperforms triplet loss when you have access to many negatives.

---

## Common Implementations

**SimCLR:**
- Batch of N images produces 2N augmented views
- Each view has 1 positive (its pair) and 2(N-1) negatives (all other views)
- Requires large batch sizes (4096+)

**MoCo:**
- Uses a momentum-updated encoder to maintain a large queue of negatives
- Queue stores 65,536 embeddings from recent batches
- Works with small batches (256)

**CLIP:**
- Matches images to text descriptions
- Uses InfoNCE with image-text pairs
- Trained on 400 million pairs

---

## Where InfoNCE Is Used

- **Self-supervised visual learning**: SimCLR, MoCo, BYOL, SwAV
- **Multimodal learning**: CLIP (images + text), ALIGN
- **Audio-visual learning**: matching audio with video frames
- **Sentence embeddings**: learning text representations
- **Reinforcement learning**: contrastive predictive coding for state representations
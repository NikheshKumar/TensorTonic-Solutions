# Embedding Head

The embedding head sits after the 24-layer backbone of EmbeddingGemma 2 and turns a sequence of output token states into the single vector that gets stored and searched. It has three parts, a mean over the sequence, a randomly initialized linear projection from the model width of 512 to the embedding width of 768, and L2 normalization. This problem covers all three, with the focus on which positions the mean is allowed to see.

## Reading one token versus averaging all of them

A sequence encoder produces $L$ output vectors, and an embedding model needs exactly one. The simplest option is to read a single position, such as a classification token at the start or the final token of a decoder. That works when the model has been trained to route a summary into that slot, and it puts the entire burden of compression on one position.

EmbeddingGemma 2 averages the output token states instead, following earlier work on embedding models built from language models. The backbone uses bidirectional attention, so every position already sees the whole input, and the mean combines all of those contextual views. Mean pooling also suits the multimodal inputs well. An image contributes 280 soft tokens in training and a video contributes 140 for each of up to 32 frames, and averaging lets every one of those positions contribute directly instead of hoping one marker token absorbed it all. The paper's ablation on vision budgets notes the flip side, which is that very long sequences can dilute a mean-pooled representation.

## The mask is the part that matters

Batched inputs have different lengths, so shorter rows are filled with padding up to the length $L$. Padding positions still produce output vectors, and those vectors can be large and point anywhere. If they enter the mean, the embedding of a sentence changes with the length of the longest sentence in its batch, which is a silent bug that no single example reveals. The paper says only that the head averages along the sequence axis. The exercise chooses to exclude padding from the mean and to count special tokens and modality markers as real tokens, since they belong to the input.

The denominator of the mean deserves a closer look. For row $b$ it is the count of real tokens, a positive scalar $c_b$. The projection is linear, so $W (S_b / c_b) = (W S_b) / c_b$, where $S_b$ is the masked sum. Normalization then divides by the norm, which also carries the factor $1 / c_b$, and the two cancel. A masked sum, a masked mean and a masked sum divided by $L$ therefore all give the same final embedding. What cannot be repaired afterwards is a padding vector inside the sum, because it changes the direction of $S_b$ and no positive rescaling undoes that.

## Why the output has unit length

The paper compares embeddings with cosine similarity, $\operatorname{sim}(x, y) = x^\top y / (\lVert x \rVert \lVert y \rVert)$. When every embedding already has unit norm, the denominator is 1 and cosine similarity is a plain dot product. Every later piece of the system relies on this. The contrastive and spread-out losses work with inner products of unit vectors, vector databases can use fast inner product search, and a stored embedding never needs its norm recomputed at query time.

The projection comes before the normalization for the same reason. Normalizing the pooled vector first and then projecting would produce vectors whose length depends on $W$, so the dot product would no longer equal the cosine. The order is pool, then project, then normalize, and each row is normalized by its own norm over the embedding axis.

## Pooling a row with a padded slot

Take one row with $L = 3$ and $d_M = 2$. The token states are $(2, 0)$, $(0, 4)$ and a padding vector $(-30, 10)$, with mask $(1, 1, 0)$. The masked sum is $(2, 4)$ and there are two real tokens, so the pooled vector is $(1, 2)$. With projection rows $(1, 1)$ and $(2, -1)$, the projected vector is $(3, 0)$, its norm is 3, and the embedding is $(1, 0)$.

Using the unmasked sum instead gives $(-28, 14)$, which projects to $(-14, -70)$ and normalizes to about $(-0.196, -0.981)$. That vector points almost opposite to the correct one, from a single padding slot. Dividing the masked sum $(2, 4)$ by $L = 3$ instead of 2 gives $(0.667, 1.333)$, which projects to $(2, 0)$ and still normalizes to $(1, 0)$, so the count only rescales.

## Common mistakes to avoid

- **Averaging over padding.** A plain mean over the sequence axis lets padding vectors steer the direction of the embedding.
- **Normalizing before the projection.** The projected vector then has a length set by $W$, so its dot products stop being cosine similarities.
- **Normalizing over the batch axis.** Each embedding must be divided by its own norm, and a norm taken across rows mixes unrelated inputs.
- **Skipping the final normalization.** The result keeps the scale of the token states and breaks every similarity computed downstream.
- **Adding a bias or a nonlinearity.** The paper's head is a plain linear projection, and either addition changes the direction of the output.
- **Reading only the first position.** A single slot ignores the rest of the sequence, which the mean is designed to use.
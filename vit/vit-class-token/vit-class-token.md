
# <span style="font-size: 20px;">[CLS] Token</span>

<span style="font-size: 14px;">The [CLS] token is a learnable embedding prepended to the sequence of patch embeddings in the Vision Transformer (ViT). After passing through all transformer layers, the state of this single token at the output serves as the image representation for classification. This idea was borrowed directly from BERT, where a similar token aggregates sentence-level information.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">In ViT, an input image is split into fixed-size patches (e.g., $16 \times 16$ pixels), each linearly projected into a $D$-dimensional embedding. This produces a sequence of $N$ patch tokens, each of shape $(1, D)$. The [CLS] token is an additional learnable parameter of shape $(1, 1, D)$ that is prepended to this sequence before the transformer encoder processes it.</span>

<span style="font-size: 14px;">The [CLS] token is not derived from any region of the image. It carries no spatial information at initialization. Instead, it starts as a randomly initialized vector and learns, through backpropagation, to aggregate information from all patches via the self-attention mechanism. By the final transformer layer, the [CLS] token's state encodes a global summary of the entire image.</span>

<span style="font-size: 14px;">The classification head (typically a small MLP) is attached exclusively to the [CLS] token's output, ignoring the patch token outputs entirely. This makes the [CLS] token the bottleneck through which all image-level information must flow.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">Let the patch embeddings after linear projection be $\mathbf{z}_1, \mathbf{z}_2, \ldots, \mathbf{z}_N$, each $\in \mathbb{R}^D$. The learnable [CLS] token is a parameter $\mathbf{x}_{\text{class}} \in \mathbb{R}^D$.</span>

<span style="font-size: 14px;">**Tiling across the batch.** The [CLS] token is a single parameter shared across all images, so it must be replicated to match the batch size $B$:</span>

$$
\mathbf{x}_{\text{class}}^{\text{tiled}} = \text{tile}(\mathbf{x}_{\text{class}}, B) \in \mathbb{R}^{B \times 1 \times D}
$$

<span style="font-size: 14px;">**Concatenation.** The tiled [CLS] token is concatenated to the front of the patch sequence along the sequence dimension:</span>

$$
\mathbf{z}_0 = [\mathbf{x}_{\text{class}}^{\text{tiled}} \;;\; \mathbf{z}_1, \mathbf{z}_2, \ldots, \mathbf{z}_N] \in \mathbb{R}^{B \times (N+1) \times D}
$$

<span style="font-size: 14px;">The sequence length changes from $N$ to $N + 1$. This augmented sequence is then passed through the transformer encoder (after adding positional embeddings).</span>

<span style="font-size: 14px;">**Output extraction.** After $L$ transformer layers, the output is $\mathbf{z}_L \in \mathbb{R}^{B \times (N+1) \times D}$. The [CLS] token's final state is extracted from position 0:</span>

$$
\mathbf{y} = \mathbf{z}_L[:, 0, :] \in \mathbb{R}^{B \times D}
$$

<span style="font-size: 14px;">This vector $\mathbf{y}$ is fed to the classification head to produce class logits.</span>

---

## <span style="font-size: 16px;">Why a [CLS] Token</span>

<span style="font-size: 14px;">Transformers produce a sequence of output tokens (one per input token), but image classification requires a single vector per image. The [CLS] token solves this aggregation problem.</span>

<span style="font-size: 14px;">**Borrowed from BERT.** Devlin et al. (2019) introduced the [CLS] token for sentence-level tasks. Dosovitskiy et al. (2020) directly adapted this for vision: "Similar to BERT's [class] token, we prepend a learnable embedding whose state at the output serves as the image representation."</span>

<span style="font-size: 14px;">**Fixed position for classification.** The [CLS] token is always at position 0, so the classification head always knows exactly where to look. There is no ambiguity about which token to use -- extraction is simply indexing position 0 of the output sequence.</span>

<span style="font-size: 14px;">**Attends to all patches through self-attention.** In each transformer layer, the [CLS] token computes attention scores against every patch token. It selectively attends to patches most informative for classification. Over multiple layers, it progressively refines its representation by gathering information from all spatial locations.</span>

<span style="font-size: 14px;">**Aggregates image-level information.** Unlike any single patch token (which is biased toward a local region), the [CLS] token is position-agnostic. It has no spatial prior and must learn to integrate global context. By the final layer, it represents a learned aggregation of the entire image, weighted by what the attention layers deemed relevant.</span>

---

## <span style="font-size: 16px;">Why Learnable</span>

<span style="font-size: 14px;">The [CLS] token is a trainable parameter, not a fixed constant. This is a deliberate design choice.</span>

<span style="font-size: 14px;">**Initialized randomly.** The [CLS] token is initialized from a normal distribution scaled by $0.02$ (i.e., `randn * 0.02`). The small scale prevents the token from dominating attention scores at the start of training, allowing gradients to flow normally.</span>

<span style="font-size: 14px;">**Learned during training.** As the model trains on classification loss, gradients flow back through the classification head, through the transformer layers, and into the [CLS] token parameter. The token learns an initialization state that, when processed by self-attention, produces the most useful image-level representation. In effect, it learns what "question" to ask the patch tokens.</span>

<span style="font-size: 14px;">**Adapts to the task.** When fine-tuning ViT on different downstream tasks, the [CLS] token adapts along with all other parameters. A fixed, non-learnable token (e.g., a zero vector) would provide no task-specific signal and would rely entirely on the transformer layers to compensate.</span>

---

## <span style="font-size: 16px;">Why Prepend, Not Append</span>

<span style="font-size: 14px;">The [CLS] token is placed at position 0 (the front of the sequence), not at the end. While self-attention is permutation-equivariant (position is encoded via positional embeddings, not by ordering), the prepend convention matters for practical reasons.</span>

<span style="font-size: 14px;">**Convention from BERT.** BERT always places [CLS] at the beginning. ViT follows this directly. Since ViT's contribution was showing that a standard transformer works for vision with minimal modifications, keeping conventions identical to NLP transformers reinforced that message.</span>

<span style="font-size: 14px;">**Position 0 means consistent extraction.** Regardless of the number of patches $N$ (which changes with image resolution or patch size), the [CLS] token is always at index 0. If the [CLS] token were appended, its position would be at index $N$, which varies with input configuration. Position 0 is the simplest, most robust convention.</span>

<span style="font-size: 14px;">**Positional embedding alignment.** The positional embedding at position 0 is specifically learned for the [CLS] token. The model learns that position 0 is special (a classification aggregator, not a spatial patch), helping the attention layers treat it differently from the patch tokens.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">The [CLS] token was introduced for vision in "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (Dosovitskiy et al., 2020).</span>

<span style="font-size: 14px;">**BERT inspiration.** The authors explicitly state: "Similar to BERT's [class] token, we prepend a learnable embedding to the sequence of embedded patches, whose state at the output of the Transformer encoder serves as the image representation." The mechanism is identical in structure to BERT's approach.</span>

<span style="font-size: 14px;">**Alternative: Global Average Pooling.** The paper also tested applying global average pooling (GAP) over the patch token outputs instead of using a [CLS] token. GAP computes the mean of all $N$ patch embeddings at the output, producing a single $D$-dimensional vector without an extra token.</span>

<span style="font-size: 14px;">**Empirical comparison.** The paper found that [CLS] and GAP perform comparably but require different learning rate schedules. When using [CLS], the learning rate schedule from BERT worked well. The paper defaults to [CLS] to stay consistent with the original transformer design.</span>

<span style="font-size: 14px;">**Pre-training and fine-tuning.** During pre-training on large datasets (JFT-300M, ImageNet-21k), the [CLS] token learns a general-purpose image representation. During fine-tuning, the classification head is replaced with a new zero-initialized linear layer, but the [CLS] token parameter is kept, providing a strong initialization.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider a small ViT with batch size $B = 2$, number of patches $N = 4$, and embedding dimension $D = 3$.</span>

### <span style="font-size: 14px;">Step 1: Define the [CLS] Token Parameter</span>

<span style="font-size: 14px;">The [CLS] token is a learnable parameter of shape $(1, 1, D) = (1, 1, 3)$, initialized with small random values (e.g., `randn * 0.02`):</span>

$$
\mathbf{x}_{\text{class}} = [[\;0.01, \;-0.02, \;0.03\;]]
$$

### <span style="font-size: 14px;">Step 2: Patch Embeddings</span>

<span style="font-size: 14px;">After splitting images into patches and projecting them, we have $N = 4$ patch embeddings per image. For a batch of $B = 2$ images, the patch embedding tensor has shape $(2, 4, 3)$:</span>

$$
\mathbf{Z}_{\text{img1}} = \begin{bmatrix} 0.5 & -0.3 & 0.1 \\ 0.2 & 0.4 & -0.1 \\ -0.2 & 0.6 & 0.3 \\ 0.1 & -0.1 & 0.5 \end{bmatrix}, \quad \mathbf{Z}_{\text{img2}} = \begin{bmatrix} -0.4 & 0.2 & 0.3 \\ 0.3 & -0.5 & 0.1 \\ 0.1 & 0.3 & -0.2 \\ -0.1 & 0.4 & 0.2 \end{bmatrix}
$$

### <span style="font-size: 14px;">Step 3: Tile the [CLS] Token Across the Batch</span>

<span style="font-size: 14px;">The single [CLS] token must be replicated $B = 2$ times so each image gets its own copy:</span>

$$
\mathbf{x}_{\text{class}}^{\text{tiled}} = \begin{bmatrix} [0.01, -0.02, 0.03] \\ [0.01, -0.02, 0.03] \end{bmatrix} \in \mathbb{R}^{2 \times 1 \times 3}
$$

<span style="font-size: 14px;">Both images receive the same [CLS] token values. Self-attention will produce different outputs per image because the patch embeddings differ. During backpropagation, gradients from all batch elements are summed to update the single shared parameter.</span>

### <span style="font-size: 14px;">Step 4: Concatenate to the Front</span>

<span style="font-size: 14px;">Concatenate along dimension 1 (sequence dimension). For image 1, the resulting sequence of shape $(5, 3)$ is:</span>

* <span style="font-size: 14px;">**Position 0 ([CLS]):** $[0.01, -0.02, 0.03]$</span>
* <span style="font-size: 14px;">**Position 1 (Patch 1):** $[0.5, -0.3, 0.1]$</span>
* <span style="font-size: 14px;">**Position 2 (Patch 2):** $[0.2, 0.4, -0.1]$</span>
* <span style="font-size: 14px;">**Position 3 (Patch 3):** $[-0.2, 0.6, 0.3]$</span>
* <span style="font-size: 14px;">**Position 4 (Patch 4):** $[0.1, -0.1, 0.5]$</span>

<span style="font-size: 14px;">The sequence length increased from $N = 4$ to $N + 1 = 5$.</span>

### <span style="font-size: 14px;">Step 5: Shape Summary</span>

* <span style="font-size: 14px;">**cls_token parameter:** $(1, 1, 3)$</span>
* <span style="font-size: 14px;">**After tiling:** $(2, 1, 3)$</span>
* <span style="font-size: 14px;">**Patch embeddings:** $(2, 4, 3)$</span>
* <span style="font-size: 14px;">**After concatenation:** $(2, 5, 3)$ -- i.e., $(B, N+1, D)$</span>

<span style="font-size: 14px;">After the transformer encoder, the classification output is extracted from position 0: $\mathbf{y} = \mathbf{z}_L[:, 0, :] \in \mathbb{R}^{2 \times 3}$.</span>

---

## <span style="font-size: 16px;">[CLS] Token vs Global Average Pooling</span>

<span style="font-size: 14px;">The [CLS] token is not the only way to aggregate patch information. Global Average Pooling (GAP) is the main alternative, and the ViT paper explicitly compared both.</span>

<span style="font-size: 14px;">**How GAP works.** Instead of prepending a [CLS] token and extracting position 0, GAP averages all $N$ patch token outputs:</span>

$$
\mathbf{y}_{\text{GAP}} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{z}_L^{(i)} \in \mathbb{R}^{B \times D}
$$

<span style="font-size: 14px;">This is parameter-free and does not increase the sequence length.</span>

<span style="font-size: 14px;">**ViT paper findings.** Both achieve similar accuracy when properly tuned, but require different hyperparameter settings, particularly learning rate schedules. The paper chose [CLS] as the default to maintain consistency with BERT.</span>

<span style="font-size: 14px;">**Trade-offs:**</span>

* <span style="font-size: 14px;">**Sequence length.** [CLS] adds one token, making self-attention $O((N+1)^2)$ vs $O(N^2)$. For typical ViT configurations ($N = 196$ for ViT-B/16), the overhead is negligible.</span>
* <span style="font-size: 14px;">**Simplicity.** GAP is simpler: no extra parameter, no tiling, no concatenation. Just a mean operation after the transformer.</span>
* <span style="font-size: 14px;">**Learned vs uniform aggregation.** The [CLS] token learns to attend selectively to informative patches through attention. GAP gives equal weight to all patches. However, the transformer layers before GAP can still modulate patch representations, so the distinction is nuanced.</span>
* <span style="font-size: 14px;">**Convention.** [CLS] is the default in ViT, DeiT, BEiT, and MAE. Some later architectures prefer GAP: Swin Transformer uses average pooling. The choice is often convention rather than a clear winner.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

### <span style="font-size: 14px;">Wrong Initialization Scale</span>

<span style="font-size: 14px;">The [CLS] token should be initialized with standard deviation $0.02$. Using standard normal (`randn` without scaling) produces values too large, destabilizing early attention scores. Zero initialization is also problematic: a zero vector produces zero attention logits, making the [CLS] token invisible to softmax. The `randn * 0.02` convention matches the initialization used for other ViT parameters.</span>

### <span style="font-size: 14px;">Forgetting to Tile Across the Batch</span>

<span style="font-size: 14px;">The [CLS] token parameter has shape $(1, 1, D)$, but the patch embeddings have shape $(B, N, D)$. Concatenating directly without tiling fails when $B > 1$ because the batch dimensions do not match. The [CLS] token must be expanded or tiled to shape $(B, 1, D)$ before concatenation. Some frameworks support broadcasting with `expand()`, but the operation must be explicit.</span>

### <span style="font-size: 14px;">Appending Instead of Prepending</span>

<span style="font-size: 14px;">The [CLS] token must go at position 0, not position $N$. Appending breaks the extraction logic (which expects position 0) and misaligns with positional embeddings. The positional embedding at position 0 was learned to correspond to [CLS] during training; placing it elsewhere means it receives the wrong positional signal.</span>

### <span style="font-size: 14px;">Wrong Sequence Length After Concatenation</span>

<span style="font-size: 14px;">After prepending, the sequence length is $N + 1$, not $N$. The positional embedding table must have $N + 1$ entries (one for [CLS] plus one per patch). Creating a table of size $N$ causes a shape mismatch when adding positional embeddings to the augmented sequence.</span>

### <span style="font-size: 14px;">Extracting the Wrong Position at Output</span>

<span style="font-size: 14px;">The classification output is the [CLS] token at position 0: `output[:, 0, :]`. Using position $-1$ (the last patch token) or position $1$ (the first patch) gives the wrong representation. The [CLS] token is always at position 0, and this is the only correct extraction index.</span>

---

# <span style="font-size: 20px;">Position Embedding</span>

<span style="font-size: 14px;">Position embeddings inject spatial location information into the token sequence of a Vision Transformer (ViT). Because self-attention is permutation-invariant -- it produces the same output regardless of token order -- the model has no inherent way to distinguish patches from different locations. Position embeddings, a core component in Dosovitskiy et al. (2020), solve this by adding a learnable vector to each token so the model can reason about spatial origin.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">A position embedding in ViT is a learnable parameter matrix $E_{\text{pos}} \in \mathbb{R}^{1 \times N \times D}$ that is added element-wise to the sequence of patch embeddings (plus the CLS token) before the sequence enters the Transformer encoder. Each row of $E_{\text{pos}}$ corresponds to one position in the sequence and holds a $D$-dimensional vector encoding "this is position $i$". Because the matrix is learned during training rather than computed from a fixed formula, the model discovers whatever positional representation is most useful for the task.</span>

<span style="font-size: 14px;">The critical detail is what $N$ represents. An image of resolution $H \times W$ is divided into non-overlapping patches of size $P \times P$, yielding $\frac{H}{P} \times \frac{W}{P}$ patch tokens. A CLS token is prepended, so $N = \frac{H}{P} \times \frac{W}{P} + 1$. The position embedding must cover all $N$ positions -- one for the CLS token and one for every patch.</span>

<span style="font-size: 14px;">The shape $(1, N, D)$ means the position embedding is shared across every image in a batch. The leading dimension of 1 allows broadcasting to replicate the same positional information across the batch dimension $B$, so a batch of shape $(B, N, D)$ receives identical position vectors regardless of batch size.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">The position embedding is applied in a single addition immediately after patch embedding and CLS token prepending:</span>

$$
z_0 = x_{\text{patches}} + E_{\text{pos}}
$$

<span style="font-size: 14px;">where:</span>

* <span style="font-size: 14px;">$x_{\text{patches}} \in \mathbb{R}^{B \times N \times D}$ is the sequence of embedded patches with the CLS token prepended.</span>
* <span style="font-size: 14px;">$E_{\text{pos}} \in \mathbb{R}^{1 \times N \times D}$ is the learnable position embedding matrix, initialized once and updated by gradient descent.</span>
* <span style="font-size: 14px;">$z_0 \in \mathbb{R}^{B \times N \times D}$ is the input to the first Transformer encoder layer, carrying both content and position information.</span>

<span style="font-size: 14px;">The addition broadcasts $E_{\text{pos}}$ from $(1, N, D)$ to $(B, N, D)$. For every image $b$, position $i$, and feature $j$:</span>

$$
z_0[b, i, j] = x_{\text{patches}}[b, i, j] + E_{\text{pos}}[0, i, j]
$$

<span style="font-size: 14px;">This is pure element-wise addition -- no concatenation, no multiplication. Position information is blended directly into the same representation space as the content information.</span>

---

## <span style="font-size: 16px;">Why Position Embeddings Are Necessary</span>

<span style="font-size: 14px;">Self-attention computes weighted sums over all tokens. The attention weight between token $i$ and token $j$ depends only on their content vectors $q_i$ and $k_j$, not on their positions. If all patch embeddings were randomly shuffled, the attention outputs would be the shuffled version of the original outputs -- the model produces the same prediction for any permutation of patches.</span>

<span style="font-size: 14px;">This property is called **permutation invariance**:</span>

$$
\text{Attention}(\pi(X)) = \pi(\text{Attention}(X))
$$

<span style="font-size: 14px;">For vision, this means the model cannot distinguish an intact image from one with randomly rearranged patches. Spatial structure carries essential information -- objects have shape, parts have relative positions, edges are continuous -- and without position encoding all of this is lost.</span>

<span style="font-size: 14px;">Adding $E_{\text{pos}}$ breaks permutation invariance. After the addition, token $i$ carries $x_i + E_{\text{pos}}[i]$ and token $j$ carries $x_j + E_{\text{pos}}[j]$. Even if $x_i = x_j$ (two identical patches), their representations differ because $E_{\text{pos}}[i] \neq E_{\text{pos}}[j]$. The attention mechanism can now differentiate based on spatial origin.</span>

---

## <span style="font-size: 16px;">1D vs 2D Position Embeddings</span>

<span style="font-size: 14px;">Patches originate from a 2D grid, so a natural question is whether position embeddings should encode 2D coordinates (row, column) rather than a flat 1D index. Dosovitskiy et al. (2020) explicitly tested both.</span>

<span style="font-size: 14px;">**1D position embeddings** assign a single learnable vector to each position in the flattened sequence. A $4 \times 4$ grid produces positions $0, 1, 2, \ldots, 15$ (plus CLS). No explicit row or column encoding exists -- the model must learn 2D structure from data.</span>

<span style="font-size: 14px;">**2D position embeddings** use two separate embedding tables: one for row index and one for column index. Each patch at grid position $(r, c)$ receives the sum of row embedding $r$ and column embedding $c$. This explicitly encodes the 2D grid and reduces unique embeddings from $\frac{H}{P} \times \frac{W}{P}$ to $\frac{H}{P} + \frac{W}{P}$.</span>

<span style="font-size: 14px;">The paper's finding: **1D and 2D position embeddings perform nearly identically**. No significant accuracy difference was observed across model sizes and datasets. The authors chose 1D for simplicity -- a single parameter matrix versus two tables with a combining step.</span>

<span style="font-size: 14px;">This result makes sense in hindsight. Learned 1D embeddings implicitly recover 2D structure during training. When visualized, nearby patches in the 2D grid have similar embedding vectors and a clear row-column pattern emerges, despite the flat parameterization.</span>

---

## <span style="font-size: 16px;">Learned vs Fixed Position Encodings</span>

<span style="font-size: 14px;">Different Transformer architectures handle position information differently. ViT's learned additive embeddings are one point in a broader design space.</span>

<span style="font-size: 14px;">**Sinusoidal (fixed) encoding** from the original Transformer (Vaswani et al., 2017) uses deterministic sine and cosine functions:</span>

$$
PE(pos, 2i) = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \quad PE(pos, 2i+1) = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)
$$

<span style="font-size: 14px;">These require no training and generalize to longer sequences, but embed a fixed inductive bias about position structure that may not be optimal for all tasks.</span>

<span style="font-size: 14px;">**Learned position embeddings** are a parameter matrix updated by backpropagation. ViT uses this approach, initializing with $\text{randn} \times 0.02$. The downside is no extrapolation: if the model trains with $N = 197$ positions, it has no embedding for position 197 or beyond.</span>

<span style="font-size: 14px;">**Rotary Position Embedding (RoPE)** (Su et al., 2021) encodes position by rotating query and key vectors so their dot product depends on relative distance $m - n$. Used in LLaMA, Mistral, and most modern LLMs.</span>

<span style="font-size: 14px;">Dosovitskiy et al. (2020) found learned and sinusoidal encodings performed equally for ViT, so they chose the simpler learned approach.</span>

---

## <span style="font-size: 16px;">The Broadcasting Mechanism</span>

<span style="font-size: 14px;">The addition $x_{\text{patches}} + E_{\text{pos}}$ relies on broadcasting -- dimensions of size 1 are automatically expanded to match the other operand. Dimension by dimension from right to left:</span>

* <span style="font-size: 14px;">**Dimension 2 (features):** Both have size $D$. Match.</span>
* <span style="font-size: 14px;">**Dimension 1 (sequence):** Both have size $N$. Match.</span>
* <span style="font-size: 14px;">**Dimension 0 (batch):** $E_{\text{pos}}$ has size 1, $x_{\text{patches}}$ has size $B$. Size-1 broadcasts: $E_{\text{pos}}$ is conceptually replicated $B$ times.</span>

<span style="font-size: 14px;">Every image in the batch receives exactly the same position embeddings. This is correct: position is a property of the grid structure, not of individual images. No memory is allocated for the replication -- broadcasting is a view operation, not a copy.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">Dosovitskiy et al. (2020) in "An Image is Worth 16x16 Words" state: "Position embeddings are added to the patch embeddings to retain positional information. We use standard learnable 1D position embeddings." The word "standard" is deliberate -- the authors position this as the simplest option, not a novel contribution. The paper's thesis is that a standard Transformer with minimal vision-specific modifications can match convolutional networks given sufficient data.</span>

<span style="font-size: 14px;">The paper provides a visualization of the learned position embeddings (Figure 7) revealing that despite 1D parameterization, the embeddings exhibit clear 2D structure: each position has highest cosine similarity with positions from the same row and column of the patch grid. The model autonomously discovers spatial layout during training.</span>

<span style="font-size: 14px;">Removing position embeddings entirely dropped accuracy from approximately 79.4% to approximately 64.2% on ImageNet for ViT-B/16 trained on ImageNet-21k -- a roughly 15-point drop confirming that position information is essential, not optional.</span>

---

## Numerical Example ($B=2, N=5, D=3$)

<span style="font-size: 14px;">Consider a batch of 2 images, each with 4 patches and a CLS token ($N = 5$), embedding dimension $D = 3$.</span>

<span style="font-size: 14px;">**Patch embeddings** $x_{\text{patches}} \in \mathbb{R}^{2 \times 5 \times 3}$:</span>

<span style="font-size: 14px;">Image 0: $[[0.10, 0.20, 0.30],\; [0.40, 0.50, 0.60],\; [0.70, 0.80, 0.90],\; [1.00, 1.10, 1.20],\; [1.30, 1.40, 1.50]]$</span>

<span style="font-size: 14px;">Image 1: $[[0.01, 0.02, 0.03],\; [0.04, 0.05, 0.06],\; [0.07, 0.08, 0.09],\; [0.10, 0.11, 0.12],\; [0.13, 0.14, 0.15]]$</span>

<span style="font-size: 14px;">**Position embedding** $E_{\text{pos}} \in \mathbb{R}^{1 \times 5 \times 3}$ (note small magnitudes from $\text{randn} \times 0.02$):</span>

<span style="font-size: 14px;">$[[0.02, -0.01, 0.03],\; [0.01, 0.02, -0.02],\; [-0.01, 0.03, 0.01],\; [0.03, -0.02, 0.02],\; [-0.02, 0.01, -0.01]]$</span>

<span style="font-size: 14px;">**Result for image 0:**</span>

* <span style="font-size: 14px;">CLS: $[0.10, 0.20, 0.30] + [0.02, -0.01, 0.03] = [0.12, 0.19, 0.33]$</span>
* <span style="font-size: 14px;">Patch 1: $[0.40, 0.50, 0.60] + [0.01, 0.02, -0.02] = [0.41, 0.52, 0.58]$</span>
* <span style="font-size: 14px;">Patch 2: $[0.70, 0.80, 0.90] + [-0.01, 0.03, 0.01] = [0.69, 0.83, 0.91]$</span>
* <span style="font-size: 14px;">Patch 3: $[1.00, 1.10, 1.20] + [0.03, -0.02, 0.02] = [1.03, 1.08, 1.22]$</span>
* <span style="font-size: 14px;">Patch 4: $[1.30, 1.40, 1.50] + [-0.02, 0.01, -0.01] = [1.28, 1.41, 1.49]$</span>

<span style="font-size: 14px;">**Result for image 1:**</span>

* <span style="font-size: 14px;">CLS: $[0.01, 0.02, 0.03] + [0.02, -0.01, 0.03] = [0.03, 0.01, 0.06]$</span>
* <span style="font-size: 14px;">Patch 1: $[0.04, 0.05, 0.06] + [0.01, 0.02, -0.02] = [0.05, 0.07, 0.04]$</span>
* <span style="font-size: 14px;">Patch 2: $[0.07, 0.08, 0.09] + [-0.01, 0.03, 0.01] = [0.06, 0.11, 0.10]$</span>
* <span style="font-size: 14px;">Patch 3: $[0.10, 0.11, 0.12] + [0.03, -0.02, 0.02] = [0.13, 0.09, 0.14]$</span>
* <span style="font-size: 14px;">Patch 4: $[0.13, 0.14, 0.15] + [-0.02, 0.01, -0.01] = [0.11, 0.15, 0.14]$</span>

<span style="font-size: 14px;">Both images received the same position offsets. The CLS token at position 0 always gets $[0.02, -0.01, 0.03]$, patch 1 always gets $[0.01, 0.02, -0.02]$, and so on. Position information is content-independent.</span>

---

## <span style="font-size: 16px;">What Position Embeddings Learn</span>

<span style="font-size: 14px;">Although initialized randomly, training shapes position embeddings into a structured representation reflecting 2D image geometry. Dosovitskiy et al. (2020) visualized this by computing cosine similarity between each pair of position embedding vectors, revealing several patterns:</span>

* <span style="font-size: 14px;">**Local similarity:** Each position embedding is most similar to embeddings of adjacent positions in the 2D grid. A patch at $(2, 3)$ has higher similarity with $(2, 2)$, $(2, 4)$, $(1, 3)$, and $(3, 3)$ than with distant patches.</span>
* <span style="font-size: 14px;">**Row-column structure:** The similarity pattern shows horizontal and vertical bands, indicating separate row and column representations emerge despite using a single 1D table.</span>
* <span style="font-size: 14px;">**CLS token distinctness:** The CLS position embedding has low similarity with all patch positions, consistent with its role as a special aggregation token.</span>
* <span style="font-size: 14px;">**Smooth gradients:** Similarity decreases gradually with distance, meaning the model encodes spatial proximity, not just position identity.</span>

<span style="font-size: 14px;">This emergent structure explains why 1D and 2D embeddings produce equivalent results: the 1D parameterization is expressive enough to recover 2D structure on its own.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Wrong value of N (forgetting CLS adds one position).** If an image produces $14 \times 14 = 196$ patches and a CLS token is prepended, $N = 197$, not 196. A position embedding of shape $(1, 196, D)$ added to a sequence of length 197 causes a shape mismatch. The CLS token at position 0 needs its own embedding vector.</span>
* <span style="font-size: 14px;">**Wrong tensor shape.** Creating $E_{\text{pos}}$ with shape $(N, D)$ instead of $(1, N, D)$ may cause incorrect broadcasting or errors depending on the framework. The canonical ViT implementation uses $(1, N, D)$ to make broadcasting semantics explicit.</span>
* <span style="font-size: 14px;">**Concatenating instead of adding.** A common error is concatenating position as additional features, producing shape $(B, N, 2D)$ instead of $(B, N, D)$. This doubles the feature dimension, breaks the input size expected by the encoder, and does not match the ViT architecture. Position embeddings are always added.</span>
* <span style="font-size: 14px;">**Not using broadcasting.** Manually tiling or repeating $E_{\text{pos}}$ $B$ times wastes memory by allocating $B$ copies of identical data. Worse, storing the tiled tensor as a parameter can cause incorrect gradient accumulation. Broadcasting avoids both issues.</span>
* <span style="font-size: 14px;">**Wrong resolution at inference.** Learned position embeddings have a fixed $N$ from the training resolution. Training on $224 \times 224$ with $P = 16$ gives $N = 197$. Feeding $384 \times 384$ at inference produces 577 patches, but the model only has 197 position embeddings. The fix is 2D bicubic interpolation of the patch grid embeddings from $14 \times 14$ to $24 \times 24$, with the CLS embedding handled separately. Forgetting this causes a crash or silent accuracy degradation.</span>
* <span style="font-size: 14px;">**Applying position embeddings after the encoder.** The purpose of position embeddings is to let attention layers use spatial information. Adding position after the encoder means all self-attention operates without spatial signal, degrading the model to an unordered bag-of-patches representation.</span>

---
# <span style="font-size: 20px;">ViT Multi-Head Self-Attention</span>

<span style="font-size: 14px;">Multi-head self-attention (MHSA) is the core mixing operation of a Vision Transformer (Dosovitskiy et al., 2020), inherited unchanged from the Transformer of Vaswani et al., 2017. It lets every patch token gather information from every other patch token through learned, content-dependent attention weights, giving ViT a global receptive field from the very first layer, in contrast to the local receptive field of a convolution.</span>

---

## <span style="font-size: 16px;">What the Block Computes</span>

<span style="font-size: 14px;">Given a batch of token sequences $x$ of shape $(B, N, D)$, where $N$ is the number of patch tokens and $D$ the model dimension, the block produces a new $(B, N, D)$ representation in which each token is a weighted blend of all token values. The bare attention operation, stripped of mask, dropout, residual, and layernorm, is:</span>

$$
\text{Attn}(x) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_h}}\right) V
$$

<span style="font-size: 14px;">Self-attention means $Q$, $K$, and $V$ all come from the same input $x$. A single fused projection produces all three at once, then they are split into $h$ heads, attention runs independently per head, and the heads are concatenated and projected back out.</span>

<span style="font-size: 14px;">The intuition is a soft, differentiable dictionary lookup. Each token emits a **query** describing what information it wants, every token exposes a **key** describing what it offers and a **value** carrying its content. The dot product of a query with each key measures relevance, the softmax turns those relevances into weights, and the output is the weighted average of values. Because the weights are computed from content rather than fixed by position, the same layer can route information differently for every input image.</span>

---

## <span style="font-size: 16px;">Step by Step</span>

<span style="font-size: 14px;">1. **Fused QKV projection**: compute $[Q \mid K \mid V] = x W_{qkv} + b_{qkv}$, where $W_{qkv} \in \mathbb{R}^{D \times 3D}$. Column-split the result into three $(B, N, D)$ tensors $Q$, $K$, $V$. Fusing the three projections into one matrix multiply is purely an efficiency choice: it is mathematically equivalent to three separate $D \times D$ projections $W_Q, W_K, W_V$ stacked side by side, but a single large matmul is faster on GPUs and is the standard implementation in libraries like timm.</span>

<span style="font-size: 14px;">2. **Split into heads**: reshape each of $Q, K, V$ from $(B, N, D)$ to $(B, N, h, d_h)$ with $d_h = D / h$, then permute to $(B, h, N, d_h)$ so each head is an independent $(N, d_h)$ matrix.</span>

<span style="font-size: 14px;">3. **Scaled scores**: for each head compute $S = Q K^\top / \sqrt{d_h}$, a $(B, h, N, N)$ tensor where $S_{ij}$ is the affinity of query token $i$ to key token $j$.</span>

<span style="font-size: 14px;">4. **Softmax**: apply a row-wise softmax over the last axis so each query's weights over the $N$ keys sum to 1.</span>

<span style="font-size: 14px;">5. **Weighted sum**: multiply the attention weights by $V$ to get the per-head context $(B, h, N, d_h)$, where each output row is a convex combination of value vectors.</span>

<span style="font-size: 14px;">6. **Merge heads**: permute back to $(B, N, h, d_h)$ and reshape to $(B, N, D)$, concatenating the heads along the feature axis.</span>

<span style="font-size: 14px;">7. **Output projection**: apply $W_o \in \mathbb{R}^{D \times D}$ and bias $b_o$ to mix information across heads, yielding the final $(B, N, D)$ output.</span>

---

## <span style="font-size: 16px;">Why Scale by $\sqrt{d_h}$</span>

<span style="font-size: 14px;">Each score $S_{ij} = q_i^\top k_j$ is a sum of $d_h$ products. If the entries of $q$ and $k$ are roughly independent with zero mean and unit variance, the dot product has variance $d_h$ and standard deviation $\sqrt{d_h}$. As $d_h$ grows the raw scores spread out, pushing the softmax into a regime where one entry dominates and the gradient through the other entries vanishes.</span>

<span style="font-size: 14px;">Dividing by $\sqrt{d_h}$ rescales the score variance back to roughly 1, keeping the softmax in a well-behaved range and preserving gradient flow. The Vaswani paper introduced this factor for exactly this reason, and ViT keeps it. The scale uses the **per-head** dimension $d_h$, not the full $D$, because the dot products are taken within each head over $d_h$ values.</span>

<span style="font-size: 14px;">The failure mode without scaling is concrete. The softmax gradient is proportional to $w_i(\delta_{ij} - w_j)$; when one weight $w_i \to 1$ and the rest $\to 0$, this product collapses to nearly zero, so almost no gradient reaches the keys and queries. Early in training, before the projections have learned anything useful, saturated attention means the layer cannot learn which tokens to attend to. Scaling keeps the initial logits near unit scale so the softmax starts soft and gradients flow to every key.</span>

---

## <span style="font-size: 16px;">Why Multiple Heads</span>

<span style="font-size: 14px;">A single attention head computes one set of weights per query, a single way of relating tokens. Splitting $D$ into $h$ heads of size $d_h$ lets the model run $h$ attention patterns in parallel, each in its own subspace. Different heads can specialize:</span>

* <span style="font-size: 14px;">One head may attend to spatially neighboring patches, recovering local texture</span>
* <span style="font-size: 14px;">Another may attend to distant patches sharing color or semantics, capturing global context</span>
* <span style="font-size: 14px;">Another may attend broadly, acting as a smoothing or averaging operation</span>

<span style="font-size: 14px;">Crucially, multi-head attention costs the same as single-head attention of width $D$, because the heads partition the dimension rather than duplicate it: total work is $h$ heads each doing $O(N^2 d_h)$, and $h \cdot d_h = D$. The output projection $W_o$ then recombines the per-head outputs, learning how to weight the contribution of each head. ViT-Base uses $h = 12$ heads with $D = 768$, so $d_h = 64$, the same per-head width as BERT. ViT-Large uses $h = 16$ heads with $D = 1024$, again giving $d_h = 64$, a per-head width that has proven robust across model scales.</span>

<span style="font-size: 14px;">The reason a single head is weaker is subtle: with one head the attention output for a token is forced to be one convex combination of values, so it can emphasize only one relational pattern at a time. Averaging over many tokens in a single distribution also blurs distinct signals together. Multiple heads let the model keep several relational patterns separate and only merge them, deliberately, in $W_o$. The ViT paper visualizes attention distance per head and shows that early-layer heads range from highly local to fully global, confirming this specialization emerges in practice.</span>

---

## <span style="font-size: 16px;">Quadratic Cost in Token Count</span>

<span style="font-size: 14px;">The score matrix $Q K^\top$ has shape $(N, N)$ per head, so computing and storing attention is $O(N^2 d_h)$ time and $O(N^2)$ memory per head, giving $O(N^2 D)$ overall. Because $N = HW/P^2$, attention cost grows with the fourth power of image side length at fixed patch size. This quadratic scaling is the central efficiency bottleneck of ViT and the reason high-resolution vision Transformers turn to windowed attention (Swin), linear-attention approximations, or token reduction.</span>

<span style="font-size: 14px;">The projections themselves are $O(N D^2)$, linear in $N$. So for short sequences the projections dominate, but as $N$ grows the $O(N^2 D)$ attention term takes over. For a $224 \times 224$ image at $P = 16$, $N = 196$, the $N \times N$ matrix is $196 \times 196$ per head, modest; at $4 \times$ the resolution $N = 784$ and the matrix is $16 \times$ larger.</span>

<span style="font-size: 14px;">This is the precise sense in which ViT trades inductive bias for compute. A convolution touches a fixed-size neighborhood and scales linearly with image area; attention touches everything and scales quadratically. The payoff is a global receptive field at every layer, so a patch in the top-left corner can directly influence one in the bottom-right after a single attention layer, something a CNN needs many stacked layers to achieve. For dense prediction at high resolution this quadratic memory term is what forces architectural changes like Swin's windows.</span>

---

## <span style="font-size: 16px;">The Softmax Step</span>

<span style="font-size: 14px;">The row-wise softmax converts raw scores into a probability distribution over keys:</span>

$$
\text{softmax}(s)_j = \frac{e^{s_j}}{\sum_{k} e^{s_k}}
$$

<span style="font-size: 14px;">Each query token $i$ thus produces a length-$N$ weight vector that is non-negative and sums to 1, so the attention output for that token is a convex combination of the $N$ value vectors. This is what makes attention a soft selection: instead of hard-picking one token, it spreads mass across all of them in proportion to relevance. In practice softmax is computed in a numerically stable way by subtracting the row max before exponentiating, which prevents overflow when scores are large and does not change the result since softmax is shift-invariant.</span>

<span style="font-size: 14px;">Because the output is a convex combination, each output vector lies in the convex hull of the value vectors and can never extrapolate beyond them. This is why attention alone, even stacked, is limited to averaging operations and needs the per-block MLP to introduce nonlinearity. In a ViT every patch can in principle pull from every other patch on the first layer, but the actual mixing is always a weighted mean of values, with the weights being the only content-dependent, learned part.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Take $B = 1$, $N = 2$ tokens, $D = 2$, single head so $d_h = 2$. Suppose after projection $Q = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$, $K = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$, $V = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$.</span>

<span style="font-size: 14px;">Scores: $Q K^\top = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$. Scale by $\sqrt{d_h} = \sqrt{2} \approx 1.414$: $\begin{pmatrix} 0.707 & 0 \\ 0 & 0.707 \end{pmatrix}$.</span>

<span style="font-size: 14px;">Row-wise softmax of $[0.707, 0]$: $e^{0.707} \approx 2.028$, $e^0 = 1$, sum $\approx 3.028$, weights $\approx [0.670, 0.330]$. By symmetry row 2 is $[0.330, 0.670]$.</span>

<span style="font-size: 14px;">Output $= WV$: row 1 $= 0.670 \cdot [1,0] + 0.330 \cdot [0,1] = [0.670, 0.330]$; row 2 $= [0.330, 0.670]$. Each token becomes a soft blend dominated by its own value but mixing in the other. The output projection $W_o, b_o$ would then transform these vectors.</span>

<span style="font-size: 14px;">Two checks make the example a useful debugging template. First, each output row's coordinates need not sum to 1; it is the attention **weights**, not the output values, that form a distribution. Second, had the scaling been omitted, the raw scores $[1, 0]$ would soften the gap less than the scaled $[0.707, 0]$, giving weights closer to $[0.731, 0.269]$; with larger $d_h$ that gap widens dramatically, which is exactly the saturation the $\sqrt{d_h}$ factor prevents. Working a tiny case like this by hand catches split-axis and softmax-axis bugs immediately.</span>

---

## <span style="font-size: 16px;">Self-Attention in ViT vs NLP</span>

<span style="font-size: 14px;">The mechanism is identical to NLP self-attention, but two things differ in vision use. First, ViT attention is **bidirectional and unmasked**: every patch sees every other patch, unlike the causal mask in a GPT decoder, because an image has no left-to-right generation order. Second, the tokens are patches, not words, so attention patterns correspond to spatial relationships rather than syntactic ones.</span>

<span style="font-size: 14px;">In the full ViT block this MHSA is wrapped with a pre-layernorm and a residual connection, $x \leftarrow x + \text{MHSA}(\text{LN}(x))$, then followed by an MLP sublayer with its own norm and residual. ViT uses Pre-LN (normalize before the sublayer) rather than the original Transformer's Post-LN, because Pre-LN gives more stable gradients and tolerates the deeper stacks and higher learning rates that vision pretraining needs. This problem isolates the bare attention core so the projection, head-split, scaling, and merge logic can be implemented and verified on its own, without the confounding effects of normalization, residuals, or the MLP.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Wrong QKV split axis.** The fused output is $(B, N, 3D)$ and must be split into three contiguous $D$-wide blocks along the last axis as $[Q \mid K \mid V]$. Splitting interleaved (every third column) or along the wrong axis silently scrambles the projection and produces nonsense without crashing.</span>
* <span style="font-size: 14px;">**Scaling by $\sqrt{D}$ instead of $\sqrt{d_h}$.** The dot product runs over the per-head dimension $d_h = D/h$, so the correct scale is $\sqrt{d_h}$. Using $\sqrt{D}$ over-scales by a factor of $\sqrt{h}$, flattening the softmax and weakening attention.</span>
* <span style="font-size: 14px;">**Softmax over the wrong axis.** The softmax must normalize over the key axis (last axis of the $(B, h, N, N)$ scores) so each query's weights sum to 1. Normalizing over the query axis instead mixes information across queries and breaks the per-token convex-combination property.</span>
* <span style="font-size: 14px;">**Forgetting to permute heads back before reshape.** Heads are split with a permute to $(B, h, N, d_h)$; merging requires permuting back to $(B, N, h, d_h)$ before the reshape to $(B, N, D)$. Reshaping directly from the permuted layout interleaves head and feature dimensions, corrupting the token vectors.</span>

---

## <span style="font-size: 16px;">Modern Variants</span>

* <span style="font-size: 14px;">**Windowed attention (Swin).** Restricts attention to local windows of patches, reducing cost from $O(N^2)$ to linear in $N$, with shifted windows to allow cross-window flow.</span>
* <span style="font-size: 14px;">**Multi-query and grouped-query attention.** Share keys and values across heads to shrink memory bandwidth, originally for fast LLM decoding but increasingly used in vision too.</span>
* <span style="font-size: 14px;">**FlashAttention.** Computes the same result without ever materializing the full $(N, N)$ score matrix, tiling the computation to fit on-chip memory and cutting both memory and time for long sequences.</span>
* <span style="font-size: 14px;">**Class-attention and register tokens.** Variants like CaiT separate patch-mixing from class-token attention, and DINOv2 adds learnable register tokens, but all build on the same QKV-softmax-projection core implemented here.</span>

---
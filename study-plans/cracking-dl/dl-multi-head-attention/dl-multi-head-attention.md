# <span style="font-size: 20px;">Multi-Head Attention</span>

<span style="font-size: 14px;">Multi-head attention (Vaswani et al., 2017) extends scaled dot-product attention by running multiple attention operations in parallel, each with its own learned projections. This allows the model to jointly attend to information from different representation subspaces at different positions - something a single attention head cannot do.</span>

---

## <span style="font-size: 16px;">Why Multiple Heads?</span>

<span style="font-size: 14px;">A single attention head computes one set of attention weights. Consider the sentence "The cat sat on the mat because it was tired." A single head attending from "it" might focus on "cat" (coreference) or "mat" (proximity). Multiple heads allow the model to capture both relationships simultaneously:</span>

- <span style="font-size: 14px;">Head 1 might learn syntactic relationships (subject-verb agreement)</span>
- <span style="font-size: 14px;">Head 2 might learn coreference (pronoun resolution)</span>
- <span style="font-size: 14px;">Head 3 might learn positional patterns (attend to adjacent tokens)</span>
- <span style="font-size: 14px;">Head 4 might learn semantic similarity</span>

<span style="font-size: 14px;">Each head operates on a lower-dimensional projection ($d_k = d_{\text{model}} / h$), so the total computation is the same as a single head with full dimensionality. Multiple heads provide richer representations at no additional computational cost.</span>

---

## <span style="font-size: 16px;">The Mechanism</span>

<span style="font-size: 14px;">Given input Q, K, V of shape $(B, n, d_{\text{model}})$:</span>

<span style="font-size: 14px;">**Step 1 - Project**: apply learned linear projections</span>

$$
Q' = QW^Q, \quad K' = KW^K, \quad V' = VW^V
$$

<span style="font-size: 14px;">where each $W \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$. In practice, these are single linear layers that project to the full $d_{\text{model}}$ dimension.</span>

<span style="font-size: 14px;">**Step 2 - Reshape into heads**: split the last dimension into $h$ heads of size $d_k$</span>

$$
Q' : (B, n, d_{\text{model}}) \to (B, h, n, d_k)
$$

<span style="font-size: 14px;">This is done by reshaping to $(B, n, h, d_k)$ and then transposing to $(B, h, n, d_k)$.</span>

<span style="font-size: 14px;">**Step 3 - Attention per head**: standard scaled dot-product attention independently for each head</span>

$$
\text{head}_i = \text{softmax}\!\left(\frac{Q'_i {K'_i}^T}{\sqrt{d_k}}\right) V'_i
$$

<span style="font-size: 14px;">**Step 4 - Concatenate and project**: reshape back to $(B, n, d_{\text{model}})$ and apply $W^O$</span>

$$
\text{output} = \text{Concat}(\text{head}_1, \dots, \text{head}_h) \cdot W^O
$$

---

## <span style="font-size: 16px;">Self-Attention vs Cross-Attention</span>

<span style="font-size: 14px;">Multi-head attention supports two modes:</span>

<span style="font-size: 14px;">**Self-attention**: Q, K, V all come from the same sequence. Each position attends to all positions in the same sequence. Used in Transformer encoders and the first attention layer of decoders.</span>

<span style="font-size: 14px;">**Cross-attention**: Q comes from one sequence (e.g., decoder), K and V come from another (e.g., encoder output). The decoder attends to encoder positions to retrieve relevant source information. Q and K/V can have different sequence lengths.</span>

<span style="font-size: 14px;">The implementation is identical - the only difference is what tensors are passed as Q, K, V.</span>

---

## <span style="font-size: 16px;">Parameter Efficiency and Variants</span>

<span style="font-size: 14px;">Standard multi-head attention has $4 d_{\text{model}}^2 + 4 d_{\text{model}}$ parameters (four linear projections with biases). Several variants reduce this:</span>

- <span style="font-size: 14px;">**Multi-Query Attention** (MQA, Shazeer 2019): share K and V projections across all heads. Only Q has per-head projections. Reduces KV-cache memory by $h\times$ during inference. Used in PaLM, Falcon</span>
- <span style="font-size: 14px;">**Grouped-Query Attention** (GQA, Ainslie et al. 2023): compromise between MHA and MQA. Groups of heads share K, V projections. Used in LLaMA 2, Mistral, Gemma</span>
- <span style="font-size: 14px;">**Low-rank projections**: replace $W^Q, W^K$ with low-rank factorizations to reduce compute. Used in Linformer</span>
- <span style="font-size: 14px;">**No output projection**: some lightweight models drop $W^O$ for efficiency, relying on the per-head projections alone</span>

<span style="font-size: 14px;">The KV-cache size during autoregressive inference is $2 \times L \times n \times h \times d_k$ (2 for K and V, L layers, n tokens, h heads). For long sequences with many layers, this dominates GPU memory, which is why MQA and GQA are critical for production LLMs.</span>

---


## <span style="font-size: 16px;">Computational Cost and Optimization</span>

<span style="font-size: 14px;">**Quadratic complexity.** Standard multi-head attention has O(n^2 * d) time and O(n^2 * h) memory for sequence length n, head dimension d, and h heads. This is the primary bottleneck for long-context models.</span>

<span style="font-size: 14px;">**Flash Attention** reorganizes the computation to avoid materializing the full n x n attention matrix, reducing memory from O(n^2) to O(n) through tiling and online softmax. It achieves wall-clock speedups of 2-4x on modern GPUs by improving memory access patterns.</span>

<span style="font-size: 14px;">**KV-Cache.** During autoregressive generation, the key and value projections for all previous tokens are cached so only the new token's Q, K, V need to be computed at each step. This reduces per-token computation from O(n * d) to O(d) but requires O(n * d * layers) memory, which becomes the bottleneck for long sequences.</span>

<span style="font-size: 14px;">**Grouped Query Attention (GQA)** shares K and V heads across multiple Q heads, reducing KV-cache memory by a factor equal to the group size. LLaMA 2 70B uses 8 KV heads shared across 64 query heads.</span>


## <span style="font-size: 16px;">Common Interview Follow-ups</span>

- <span style="font-size: 14px;">**Q: Why not just use a single head with larger d_k?**</span>
  <span style="font-size: 14px;">A: A single head with $d_k = d_{\text{model}}$ has the same parameter count but can only compute one attention pattern per position. Multiple heads compute $h$ different attention patterns simultaneously. Empirically, this captures richer relationships - Vaswani et al. showed that removing heads degrades performance even with the same total dimension.</span>

- <span style="font-size: 14px;">**Q: What is the computational complexity of multi-head attention?**</span>
  <span style="font-size: 14px;">A: The four projections are $O(n \cdot d^2)$. The attention itself is $O(n^2 \cdot d)$ per head, but since $d_k = d/h$ and there are $h$ heads, the total is still $O(n^2 \cdot d)$. The bottleneck is the $n^2$ term from the score matrix.</span>

- <span style="font-size: 14px;">**Q: How does grouped-query attention reduce memory?**</span>
  <span style="font-size: 14px;">A: In standard MHA, the KV-cache stores separate K and V for each head. With GQA using $g$ groups, you store K, V for only $g$ groups instead of $h$ heads. The query heads within each group share the same K, V. LLaMA 2 70B uses 8 KV groups for 64 query heads, reducing KV-cache by 8x.</span>

- <span style="font-size: 14px;">**Q: How is the mask shaped for multi-head attention?**</span>
  <span style="font-size: 14px;">A: The mask is typically $(1, 1, n_q, n_k)$ or $(B, 1, n_q, n_k)$, broadcasting across the head dimension. A causal mask is a lower-triangular matrix of ones. Padding masks have zeros for padding positions. Both are combined with logical AND for decoder self-attention.</span>

- <span style="font-size: 14px;">**Q: What happens to attention patterns in deeper layers?**</span>
  <span style="font-size: 14px;">A: Research (Clark et al., 2019) shows that early layers attend to local context and punctuation, middle layers capture syntactic relationships, and later layers attend to task-specific patterns. Many heads in deep layers become redundant, which is why head pruning (Michel et al., 2019) can remove 20-40% of heads with minimal quality loss.</span>

---
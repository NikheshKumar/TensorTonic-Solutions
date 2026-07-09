# <span style="font-size: 20px;">Multi-Head Attention</span>

<span style="font-size: 14px;">Multi-head attention is the mechanism that allows a Transformer to jointly attend to information from different representation subspaces at different positions. Instead of performing a single attention function over $d_{\text{model}}$-dimensional keys, values, and queries, the model linearly projects them $h$ times with different learned projections, runs attention in parallel on each projection, and concatenates the results. Introduced in Vaswani et al. (2017), it is the central building block of every Transformer encoder and decoder layer.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">Multi-head attention splits the representation space into $h$ independent subspaces, applies scaled dot-product attention within each subspace, and recombines the results. Each subspace is called an **attention head**. Each head learns its own set of projection weights, so different heads can learn to attend to different types of relationships: one head might focus on syntactic structure, another on coreference, another on positional proximity.</span>

<span style="font-size: 14px;">The critical insight is that this costs roughly the same as a single attention operation over the full $d_{\text{model}}$ dimension. Single-head attention with $d_k = d_{\text{model}}$ requires $O(n^2 \cdot d_{\text{model}})$ computation. Multi-head attention with $h$ heads of dimension $d_k = d_{\text{model}} / h$ requires $h \cdot O(n^2 \cdot d_k) = O(n^2 \cdot d_{\text{model}})$, the same total. The dimensionality reduction per head is exactly compensated by the number of heads.</span>

<span style="font-size: 14px;">The final step is a learned output projection $W^O$ that mixes information across heads. Without $W^O$, the heads cannot communicate with each other. The output projection allows the model to combine the diverse patterns discovered by individual heads into a single coherent representation.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">Given queries $Q$, keys $K$, and values $V$ (each of shape $n \times d_{\text{model}}$ for a sequence of $n$ tokens), multi-head attention computes:</span>

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \, W^O
$$

<span style="font-size: 14px;">where each head is a scaled dot-product attention on linearly projected inputs:</span>

$$
\text{head}_i = \text{Attention}(Q W_i^Q, \; K W_i^K, \; V W_i^V)
$$

<span style="font-size: 14px;">and the attention function within each head is:</span>

$$
\text{Attention}(Q', K', V') = \text{softmax}\!\left(\frac{Q' K'^T}{\sqrt{d_k}}\right) V'
$$

<span style="font-size: 14px;">The projection matrices and their shapes:</span>

* <span style="font-size: 14px;">**$W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$:** Projects queries into head $i$'s query subspace.</span>
* <span style="font-size: 14px;">**$W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$:** Projects keys into head $i$'s key subspace.</span>
* <span style="font-size: 14px;">**$W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$:** Projects values into head $i$'s value subspace.</span>
* <span style="font-size: 14px;">**$W^O \in \mathbb{R}^{h \cdot d_v \times d_{\text{model}}}$:** Maps the concatenated heads back to $d_{\text{model}}$.</span>
* <span style="font-size: 14px;">**$d_k = d_v = d_{\text{model}} / h$:** Each head operates on a reduced dimension. For the base Transformer, $d_{\text{model}} = 512$, $h = 8$, so $d_k = d_v = 64$.</span>

<span style="font-size: 14px;">In practice, the $h$ separate projection matrices $W_i^Q$ are stored as a single weight matrix $W^Q \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$, and the split into heads is performed by reshaping after the matrix multiplication. The same applies to $W^K$ and $W^V$. The implementation uses three large matrix multiplications followed by a reshape, rather than $3h$ smaller ones.</span>

---

## <span style="font-size: 16px;">Why Multiple Heads</span>

<span style="font-size: 14px;">A single attention head computes one set of attention weights per query position. Each query can only produce one distribution over keys, which limits the function to attending to one "type" of relationship at a time. If a word needs to simultaneously attend to its syntactic head, its coreference antecedent, and the nearest punctuation, a single head must compress all of these into a single weighted average.</span>

<span style="font-size: 14px;">Multi-head attention resolves this by giving the model $h$ independent attention patterns. Each head has its own $W_i^Q$, $W_i^K$, $W_i^V$ projections, so each head can focus on a different aspect of the input. Vaswani et al. (2017) state: "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this."</span>

<span style="font-size: 14px;">Empirical analysis confirms this. Clark et al. (2019) showed that in BERT, individual heads specialize: some attend to the next or previous token, some attend to separator tokens, some track syntactic dependency arcs, and some distribute attention broadly. This specialization emerges purely from training, not from any structural constraint beyond the separation into heads.</span>

---

## <span style="font-size: 16px;">The Reshape: Splitting Into Heads</span>

<span style="font-size: 14px;">In practice, multi-head attention does not perform $h$ separate matrix multiplications for the projections. Instead, it performs one large projection per Q, K, V and then reshapes the result to separate the heads.</span>

<span style="font-size: 14px;">The concrete steps for the query projection (keys and values follow the same pattern):</span>

<span style="font-size: 14px;">1. **Project:** Multiply $Q \in \mathbb{R}^{n \times d_{\text{model}}}$ by $W^Q \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$ to get $Q' \in \mathbb{R}^{n \times d_{\text{model}}}$.</span>

<span style="font-size: 14px;">2. **Reshape:** View $Q'$ as $\mathbb{R}^{n \times h \times d_k}$, splitting the last dimension into $h$ heads of size $d_k$.</span>

<span style="font-size: 14px;">3. **Transpose:** Permute to $\mathbb{R}^{h \times n \times d_k}$, so the head dimension comes first. Now each head's query matrix is $Q'_i \in \mathbb{R}^{n \times d_k}$.</span>

<span style="font-size: 14px;">The reshape order matters critically. The view operation must split $d_{\text{model}}$ as $(h, d_k)$, not $(d_k, h)$. If the split is reversed, each head receives interleaved features from all heads rather than a contiguous slice of the feature space. The model may still train, but pretrained weights would produce garbage output.</span>

<span style="font-size: 14px;">After attention, the reverse operation (transpose back, then reshape to merge heads) reconstructs a tensor of shape $\mathbb{R}^{n \times d_{\text{model}}}$ by concatenating the $h$ head outputs along the feature dimension.</span>

---

## <span style="font-size: 16px;">Per-Head Attention</span>

<span style="font-size: 14px;">Each head independently runs the same scaled dot-product attention. For head $i$, the projected queries $Q_i' \in \mathbb{R}^{n \times d_k}$, keys $K_i' \in \mathbb{R}^{n \times d_k}$, and values $V_i' \in \mathbb{R}^{n \times d_v}$ are used to compute:</span>

$$
\text{head}_i = \text{softmax}\!\left(\frac{Q_i' K_i'^T}{\sqrt{d_k}}\right) V_i'
$$

<span style="font-size: 14px;">The scaling by $\sqrt{d_k}$ prevents dot products from growing too large in magnitude. Without scaling, large $d_k$ causes dot products to push the softmax into regions of extremely small gradients, slowing training. Since each head uses $d_k = d_{\text{model}} / h$ rather than the full $d_{\text{model}}$, the individual dot products are naturally smaller, but the $\sqrt{d_k}$ scaling is still applied for consistent gradient behavior.</span>

<span style="font-size: 14px;">The output of each head is $\text{head}_i \in \mathbb{R}^{n \times d_v}$. For the base Transformer with $d_v = 64$, each head produces a 64-dimensional output per token position. The heads run independently with no information sharing during the attention computation.</span>

---

## <span style="font-size: 16px;">Concatenation and Output Projection</span>

<span style="font-size: 14px;">After all $h$ heads compute their attention outputs, the results are concatenated along the feature dimension and projected through $W^O$:</span>

$$
\text{output} = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \, W^O
$$

<span style="font-size: 14px;">The concatenation produces a matrix of shape $\mathbb{R}^{n \times (h \cdot d_v)} = \mathbb{R}^{n \times d_{\text{model}}}$. The output projection $W^O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$ maps this back to $d_{\text{model}}$ dimensions.</span>

<span style="font-size: 14px;">$W^O$ serves a crucial role: it mixes information across heads. Without it, the output at each position is a simple concatenation of independently computed vectors, each in its own subspace. $W^O$ enables linear combinations across these subspaces, so the model can combine, for example, syntactic attention from head 3 with semantic attention from head 7 into a single unified representation.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">In "Attention Is All You Need" (Vaswani et al., 2017), the base Transformer uses $h = 8$ parallel attention heads with $d_k = d_v = 64$ for $d_{\text{model}} = 512$. The big Transformer uses $h = 16$ with $d_k = d_v = 64$ for $d_{\text{model}} = 1024$.</span>

<span style="font-size: 14px;">The paper's ablation study (Table 3) tested different configurations. Reducing from $h = 8$ to $h = 1$ (single head) while keeping total computation constant degraded BLEU by 0.9 on English-to-German translation. Increasing to $h = 16$ or $h = 32$ with proportionally smaller $d_k$ showed diminishing returns, with $h = 32$ ($d_k = 16$) slightly worse than $h = 8$ ($d_k = 64$). This suggests each head needs sufficient dimensionality to learn useful attention patterns.</span>

<span style="font-size: 14px;">Multi-head attention appears three times in each Transformer layer: encoder self-attention, decoder self-attention (with causal masking), and encoder-decoder cross-attention. The mechanism is identical in all three cases; only the source of Q, K, V differs. In self-attention, all three come from the same sequence. In cross-attention, Q comes from the decoder and K, V come from the encoder output.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider $d_{\text{model}} = 4$, $h = 2$ heads, so $d_k = d_v = 2$. A sequence of $n = 2$ tokens.</span>

<span style="font-size: 14px;">**Input (2 tokens, 4 features):**</span>

$$
X = \begin{pmatrix} 1.0 & 0.0 & 1.0 & 0.0 \\ 0.0 & 1.0 & 0.0 & 1.0 \end{pmatrix}
$$

<span style="font-size: 14px;">**Step 1: Linear projections.** For self-attention, $Q = K = V = X$. Multiply by $W^Q \in \mathbb{R}^{4 \times 4}$:</span>

$$
W^Q = \begin{pmatrix} 0.1 & 0.2 & 0.3 & 0.4 \\ 0.5 & 0.6 & 0.7 & 0.8 \\ 0.2 & 0.1 & 0.4 & 0.3 \\ 0.6 & 0.5 & 0.8 & 0.7 \end{pmatrix}
$$

$$
Q' = X \cdot W^Q = \begin{pmatrix} 0.3 & 0.3 & 0.7 & 0.7 \\ 1.1 & 1.1 & 1.5 & 1.5 \end{pmatrix}
$$

<span style="font-size: 14px;">For brevity, assume $K' = Q'$ and $V' = Q'$ (same projections).</span>

<span style="font-size: 14px;">**Step 2: Reshape into heads.** Split $Q' \in \mathbb{R}^{2 \times 4}$ into two heads of $d_k = 2$:</span>

$$
Q'_{\text{head 1}} = \begin{pmatrix} 0.3 & 0.3 \\ 1.1 & 1.1 \end{pmatrix}, \quad Q'_{\text{head 2}} = \begin{pmatrix} 0.7 & 0.7 \\ 1.5 & 1.5 \end{pmatrix}
$$

<span style="font-size: 14px;">**Step 3: Per-head attention (Head 1).** Compute scores, scale, softmax, weighted sum:</span>

$$
S_1 = Q'_1 K'^T_1 = \begin{pmatrix} 0.18 & 0.66 \\ 0.66 & 2.42 \end{pmatrix}
$$

<span style="font-size: 14px;">Scale by $\sqrt{d_k} = \sqrt{2} \approx 1.414$:</span>

$$
S_1 / \sqrt{2} = \begin{pmatrix} 0.127 & 0.467 \\ 0.467 & 1.712 \end{pmatrix}
$$

<span style="font-size: 14px;">Row-wise softmax. Row 1: $e^{0.127} = 1.135$, $e^{0.467} = 1.595$, sum $= 2.730$, so $(0.416, 0.584)$. Row 2: $e^{0.467} = 1.595$, $e^{1.712} = 5.541$, sum $= 7.136$, so $(0.224, 0.776)$.</span>

$$
\text{head}_1 = \begin{pmatrix} 0.416 & 0.584 \\ 0.224 & 0.776 \end{pmatrix} \begin{pmatrix} 0.3 & 0.3 \\ 1.1 & 1.1 \end{pmatrix} = \begin{pmatrix} 0.767 & 0.767 \\ 0.921 & 0.921 \end{pmatrix}
$$

<span style="font-size: 14px;">Head 2 follows the same pattern with $Q'_2$, $K'_2$, $V'_2$, producing different weights because the projected values differ.</span>

<span style="font-size: 14px;">**Step 4: Concatenate and project.** Stack head outputs to get $\mathbb{R}^{2 \times 4}$, then multiply by $W^O \in \mathbb{R}^{4 \times 4}$ to produce the final output. The output projection mixes features across heads, allowing the model to combine patterns from each head.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Wrong reshape order when splitting into heads.** The view/reshape must split $d_{\text{model}}$ as $(h, d_k)$, not $(d_k, h)$. If the dimensions are transposed, each head receives interleaved features from all intended heads rather than a contiguous block. The tensor has the same shape either way, so no runtime error occurs, but the model produces incorrect results. With pretrained weights, the output is garbage.</span>
* <span style="font-size: 14px;">**Forgetting the output projection $W^O$.** Omitting $W^O$ after concatenation means heads cannot share information. The output becomes a simple interleaving of independent attention results. Performance degrades significantly because the model loses its ability to combine diverse attention patterns into a unified representation.</span>
* <span style="font-size: 14px;">**Dimension mismatch after concatenation.** The concatenated output has shape $n \times (h \cdot d_v)$. If $h \cdot d_v \neq d_{\text{model}}$ due to incorrect $d_k$ computation, the multiplication with $W^O$ crashes with a shape error. This commonly happens when $d_{\text{model}}$ is not evenly divisible by $h$.</span>
* <span style="font-size: 14px;">**Using $d_{\text{model}}$ instead of $d_k$ for the scaling factor.** Each head should scale by $\sqrt{d_k}$, not $\sqrt{d_{\text{model}}}$. Since $d_k = d_{\text{model}} / h$, using the full dimension over-scales the scores, pushing softmax outputs closer to uniform and destroying the attention mechanism's ability to focus.</span>
* <span style="font-size: 14px;">**Forgetting to transpose back after attention.** After computing per-head outputs (shape $h \times n \times d_v$), the tensor must be transposed to $n \times h \times d_v$ before reshaping to $n \times d_{\text{model}}$. Skipping the transpose merges the wrong dimensions, shuffling features across positions and heads.</span>
* <span style="font-size: 14px;">**Applying the causal mask before splitting into heads.** In decoder self-attention, the causal mask must be applied to attention scores after the $QK^T$ computation inside each head, not to the input before projection. Masking the input tensor directly destroys information and produces incorrect hidden states.</span>

---
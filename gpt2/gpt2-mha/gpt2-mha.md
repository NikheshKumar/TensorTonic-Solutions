# <span style="font-size: 20px;">Multi-Head Attention</span>

<span style="font-size: 14px;">Multi-head attention runs $h$ parallel attention heads, each operating on a smaller slice of the representation. Each head learns different relationships -- syntactic dependencies, semantic similarity, positional proximity. The outputs are concatenated and projected back to the original dimension.</span>

<span style="font-size: 14px;">In GPT-2, multi-head attention is causal: each token can only attend to itself and earlier positions. GPT-2 Small uses 12 heads with $d = 768$, giving $d_k = 64$ per head. The full pipeline: project Q, K, V, reshape into per-head tensors, compute causal attention per head, concatenate, and apply the output projection.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">Single-head attention computes one set of attention weights over the full $d$ dimensions, compressing all notions of relevance into one weighted average. Multi-head attention splits into $h$ independent subspaces and runs attention separately in each.</span>

<span style="font-size: 14px;">Each head $i$ has its own $W_q^{(i)}, W_k^{(i)}, W_v^{(i)} \in \mathbb{R}^{d \times d_k}$ where $d_k = d/h$. In practice, these are one large $W_q \in \mathbb{R}^{d \times d}$ followed by a reshape -- equivalent but GPU-efficient. Different heads learn different patterns without supervision. Trained Transformers show heads specializing in:</span>

* <span style="font-size: 14px;">**Syntactic structure:** Subject-verb agreement across clauses, dependency parsing patterns.</span>
* <span style="font-size: 14px;">**Positional patterns:** Attending to the previous token, or to specific relative positions.</span>
* <span style="font-size: 14px;">**Semantic similarity:** Grouping tokens with related meanings regardless of position.</span>
* <span style="font-size: 14px;">**Copy/induction:** Recognizing repeated patterns and predicting the next token in a sequence.</span>

<span style="font-size: 14px;">Each head defines its own "query-key similarity" through its Q and K projections, so the model distributes different linguistic functions across heads to minimize loss.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">**Step 1 -- Project Q, K, V from the input:**</span>

$$
Q = x \cdot W_q, \quad K = x \cdot W_k, \quad V = x \cdot W_v
$$

<span style="font-size: 14px;">where $x \in \mathbb{R}^{B \times S \times d}$ is the input, and $W_q, W_k, W_v \in \mathbb{R}^{d \times d}$. Each projection produces a tensor of shape $(B, S, d)$.</span>

<span style="font-size: 14px;">**Step 2 -- Reshape and transpose into per-head tensors:**</span>

$$
Q_{\text{heads}} = Q.\text{view}(B, S, h, d_k).\text{transpose}(1, 2) \implies (B, h, S, d_k)
$$

<span style="font-size: 14px;">The same reshape applies to $K$ and $V$. This splits the last dimension $d$ into $h$ heads of size $d_k = d/h$, then moves the head dimension before the sequence dimension so each head's data is contiguous.</span>

<span style="font-size: 14px;">**Step 3 -- Scaled dot-product attention per head with causal mask:**</span>

$$
\text{scores} = \frac{Q_{\text{heads}} \cdot K_{\text{heads}}^T}{\sqrt{d_k}}
$$

$$
\text{scores} = \text{scores} + M
$$

$$
\text{attn\_weights} = \text{softmax}(\text{scores}, \text{dim}=-1)
$$

$$
\text{attn\_output} = \text{attn\_weights} \cdot V_{\text{heads}}
$$

<span style="font-size: 14px;">where $M$ is the causal mask: an upper-triangular matrix filled with $-\infty$. The scores tensor has shape $(B, h, S, S)$ -- for each head independently, every query position computes a similarity score against every key position. After masking and softmax, the attention weights sum to 1 across the key dimension for each query position.</span>

<span style="font-size: 14px;">**Step 4 -- Concatenate heads and apply output projection:**</span>

$$
\text{concat} = \text{attn\_output}.\text{transpose}(1, 2).\text{contiguous}().\text{view}(B, S, d)
$$

$$
\text{output} = \text{concat} \cdot W_o
$$

<span style="font-size: 14px;">where $W_o \in \mathbb{R}^{d \times d}$. The transpose reverses Step 2, moving the head dimension back after the sequence dimension, and the view merges $h \times d_k$ back into $d$. The output projection mixes information across heads.</span>

---

## <span style="font-size: 16px;">Why Multiple Heads</span>

<span style="font-size: 14px;">A single attention head with dimension $d$ computes one attention pattern per token pair. This is limiting because language has many simultaneous relationships between tokens. Consider "The cat that I saw yesterday was sleeping." The word "was" needs to attend to "cat" for subject-verb agreement and to "sleeping" for predicate completion, while also tracking the relative clause structure.</span>

<span style="font-size: 14px;">With $h$ heads, the model maintains $h$ different attention distributions simultaneously. Each head's Q and K projections define a different similarity metric. Head 1 might project tokens so syntactically related words score high; Head 2 so semantically related words score high. The output projection $W_o$ learns to combine information from all heads into a single representation.</span>

<span style="font-size: 14px;">This is more powerful than making one head larger. A single head still produces one attention distribution per query. Multiple smaller heads produce $h$ different distributions. The total parameter count is identical ($3d^2$ for Q/K/V plus $d^2$ for $W_o$), but representational capacity is greater. Vaswani et al. (2017) noted that reducing $d_k$ proportionally keeps cost constant: $h$ heads times $O(S^2 \cdot d_k)$ equals $O(S^2 \cdot d)$. You get diversity for free.</span>

---

## <span style="font-size: 16px;">The Reshape Operation</span>

<span style="font-size: 14px;">The reshape from $(B, S, d)$ to $(B, h, S, d_k)$ is the most error-prone step in implementing multi-head attention. It involves two operations that must happen in the correct order.</span>

<span style="font-size: 14px;">**Step 1 -- view/reshape:** Split the last dimension $d$ into $h$ and $d_k$:</span>

$$
(B, S, d) \xrightarrow{\text{view}} (B, S, h, d_k)
$$

<span style="font-size: 14px;">This is a zero-copy operation. The last $d$ values per token are reinterpreted as $h$ groups of $d_k$ values. With $d = 4$, $h = 2$: values $[v_0, v_1, v_2, v_3]$ become head 0 = $[v_0, v_1]$ and head 1 = $[v_2, v_3]$.</span>

<span style="font-size: 14px;">**Step 2 -- transpose:** Swap the sequence and head dimensions:</span>

$$
(B, S, h, d_k) \xrightarrow{\text{transpose}(1,2)} (B, h, S, d_k)
$$

<span style="font-size: 14px;">Now each $(S, d_k)$ slice is one head's complete view of the sequence. Matrix multiplications along the last two dimensions operate per-head, and PyTorch batches across $B$ and $h$ automatically.</span>

<span style="font-size: 14px;">**Why this order matters:** If you reshape with the wrong dimension ordering (e.g., `.view(B, h, S, d_k)` directly), consecutive elements in $d$ get split across heads incorrectly. Each "head" would contain interleaved values from all actual heads -- semantically garbage.</span>

<span style="font-size: 14px;">After attention, the reverse operation reconstructs the original shape: transpose $(B, h, S, d_k)$ back to $(B, S, h, d_k)$, call `.contiguous()` (transpose makes memory non-contiguous), then view to $(B, S, d)$.</span>

---

## <span style="font-size: 16px;">Per-Head Causal Attention</span>

<span style="font-size: 14px;">Each head computes the same scaled dot-product attention formula independently. For head $i$, with $Q_i, K_i \in \mathbb{R}^{B \times S \times d_k}$ and $V_i \in \mathbb{R}^{B \times S \times d_k}$:</span>

$$
\text{head}_i = \text{softmax}\!\left(\frac{Q_i K_i^T}{\sqrt{d_k}} + M\right) V_i
$$

<span style="font-size: 14px;">The causal mask $M \in \mathbb{R}^{S \times S}$ is shared across all heads. It is upper-triangular (excluding diagonal) filled with $-\infty$, and 0 elsewhere. Softmax maps $-\infty$ to 0, ensuring position $t$ only attends to positions $\leq t$.</span>

<span style="font-size: 14px;">The scaling factor $\sqrt{d_k}$ prevents dot products from growing with dimension. Without scaling, large dot products push softmax into saturated regions where gradients vanish. Dividing by $\sqrt{d_k}$ keeps score variance near 1.</span>

<span style="font-size: 14px;">In implementation, heads are not computed in a loop. With shape $(B, h, S, d_k)$, the matmul `Q @ K.transpose(-2, -1)` broadcasts across both batch and head dimensions. The GPU computes all $B \times h$ attention matrices in parallel.</span>

---

## <span style="font-size: 16px;">Concat and Project</span>

<span style="font-size: 14px;">After all heads have computed their attention outputs, the results must be merged back into a single representation per token. This happens in two steps.</span>

<span style="font-size: 14px;">**Concatenation:** The per-head outputs $(B, h, S, d_k)$ are transposed and reshaped to $(B, S, d)$, merging the head and $d_k$ dimensions:</span>

$$
\text{concat}(t) = [\text{head}_1(t); \text{head}_2(t); \ldots; \text{head}_h(t)] \in \mathbb{R}^d
$$

<span style="font-size: 14px;">**Output projection:** $\text{output}(t) = \text{concat}(t) \cdot W_o$ where $W_o \in \mathbb{R}^{d \times d}$. This is not optional -- without it, each output dimension is determined by exactly one head with no cross-head mixing. $W_o$ allows the model to combine what different heads extracted (e.g., head 3 found the subject, head 7 found a modifier) into the same output dimensions.</span>

<span style="font-size: 14px;">The output shape is $(B, S, d)$ -- same as the input -- which is essential for the residual connection.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">Multi-head attention was introduced in "Attention Is All You Need" (Vaswani et al., 2017). The original Transformer used $h = 8$ heads with $d = 512$, giving $d_k = 64$. The paper demonstrated that multiple heads consistently outperformed a single head with the same total dimension.</span>

<span style="font-size: 14px;">GPT-2 (Radford et al., 2019) scaled multi-head attention across four model sizes:</span>

* <span style="font-size: 14px;">**GPT-2 Small:** $d = 768$, $h = 12$, $d_k = 64$, 12 layers, 117M parameters</span>
* <span style="font-size: 14px;">**GPT-2 Medium:** $d = 1024$, $h = 16$, $d_k = 64$, 24 layers, 345M parameters</span>
* <span style="font-size: 14px;">**GPT-2 Large:** $d = 1280$, $h = 20$, $d_k = 64$, 36 layers, 774M parameters</span>
* <span style="font-size: 14px;">**GPT-2 XL:** $d = 1600$, $h = 25$, $d_k = 64$, 48 layers, 1.5B parameters</span>

<span style="font-size: 14px;">Notice $d_k = 64$ is constant across all sizes -- more heads rather than larger heads. GPT-2 differs from the original Transformer: pre-norm (LayerNorm before attention), decoder-only (causal mask always applied, no cross-attention), and learned positional embeddings instead of sinusoidal.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Let $B = 1$, $S = 3$, $d = 4$, $h = 2$, so $d_k = 2$. We trace the full multi-head attention pipeline.</span>

<span style="font-size: 14px;">**Input** $x$ with shape $(1, 3, 4)$:</span>

$$
x = \begin{bmatrix} 1.0 & 0.0 & 1.0 & 0.0 \\ 0.0 & 1.0 & 0.0 & 1.0 \\ 1.0 & 1.0 & 0.0 & 0.0 \end{bmatrix}
$$

<span style="font-size: 14px;">**Step 1 -- Q, K, V projection.** Using $W_q = W_k = W_v = I_4$ (identity) for clarity, so $Q = K = V = x$. Each has shape $(1, 3, 4)$.</span>

<span style="font-size: 14px;">**Step 2 -- Reshape to $(B, h, S, d_k) = (1, 2, 3, 2)$:**</span>

<span style="font-size: 14px;">First view as $(1, 3, 2, 2)$, then transpose dims 1 and 2:</span>

$$
Q_{\text{head0}} = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \\ 1.0 & 1.0 \end{bmatrix}, \quad Q_{\text{head1}} = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \\ 0.0 & 0.0 \end{bmatrix}
$$

<span style="font-size: 14px;">Same split for $K$ and $V$ (since $Q = K = V = x$).</span>

<span style="font-size: 14px;">**Step 3 -- Attention for Head 0.** Compute scores with $d_k = 2$, so $\sqrt{d_k} = \sqrt{2} \approx 1.414$:</span>

$$
Q_0 K_0^T = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 1 & 1 & 2 \end{bmatrix}
$$

$$
\text{scores}_0 = \frac{1}{1.414} \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 1 & 1 & 2 \end{bmatrix} = \begin{bmatrix} 0.707 & 0 & 0.707 \\ 0 & 0.707 & 0.707 \\ 0.707 & 0.707 & 1.414 \end{bmatrix}
$$

<span style="font-size: 14px;">Apply causal mask (set upper triangle to $-\infty$):</span>

$$
\text{masked}_0 = \begin{bmatrix} 0.707 & -\infty & -\infty \\ 0 & 0.707 & -\infty \\ 0.707 & 0.707 & 1.414 \end{bmatrix}
$$

<span style="font-size: 14px;">Softmax row-wise:</span>

$$
\text{attn}_0 = \begin{bmatrix} 1.000 & 0 & 0 \\ 0.330 & 0.670 & 0 \\ 0.248 & 0.248 & 0.503 \end{bmatrix}
$$

$$
\text{out}_0 = \text{attn}_0 \cdot V_0 = \begin{bmatrix} 1.000 & 0.000 \\ 0.330 & 0.670 \\ 0.752 & 0.752 \end{bmatrix}
$$

<span style="font-size: 14px;">**Attention for Head 1.** Same process: $Q_1 K_1^T$ has diagonal structure since head 1 columns are $[1,0],[0,1],[0,0]$. After scaling, masking, and softmax:</span>

$$
\text{attn}_1 = \begin{bmatrix} 1.000 & 0 & 0 \\ 0.330 & 0.670 & 0 \\ 0.333 & 0.333 & 0.333 \end{bmatrix}, \quad \text{out}_1 = \begin{bmatrix} 1.000 & 0.000 \\ 0.330 & 0.670 \\ 0.333 & 0.333 \end{bmatrix}
$$

<span style="font-size: 14px;">**Step 4 -- Concat.** Merge heads back: transpose to $(1, 3, 2, 2)$ and view as $(1, 3, 4)$:</span>

$$
\text{concat} = \begin{bmatrix} 1.000 & 0.000 & 1.000 & 0.000 \\ 0.330 & 0.670 & 0.330 & 0.670 \\ 0.752 & 0.752 & 0.333 & 0.333 \end{bmatrix}
$$

<span style="font-size: 14px;">**Output projection.** With $W_o = I_4$ (identity for clarity): output = concat. In practice, $W_o$ would mix across head outputs, producing different values.</span>

---

## <span style="font-size: 16px;">Modern Variants</span>

<span style="font-size: 14px;">Standard multi-head attention (MHA) uses separate K and V projections per head. This creates a memory bottleneck during inference because all $h$ heads' keys and values must be cached. Three major variants address this.</span>

<span style="font-size: 14px;">**Multi-Query Attention (MQA)** (Shazeer, 2019): All heads share a single K and V. Each head has its own Q. KV cache shrinks by $h\times$, trading some quality for faster inference.</span>

<span style="font-size: 14px;">**Grouped Query Attention (GQA)** (Ainslie et al., 2023): The $h$ query heads are divided into $g$ groups sharing K/V. With $g = 1$ this is MQA; $g = h$ is standard MHA. LLaMA 2 70B uses 64 query heads with 8 KV heads. GQA recovers most of MHA quality with most of MQA speed.</span>

<span style="font-size: 14px;">**Multi-head Latent Attention (MLA)** (DeepSeek-V2, 2024): Caches a single low-rank compressed latent instead of separate K/V per head. K and V are reconstructed via learned up-projections during inference. Compresses KV cache further than MQA while matching MHA quality.</span>

---

## <span style="font-size: 16px;">Common Pitfalls</span>

* <span style="font-size: 14px;">**Wrong reshape/transpose order:** `.view(B, h, S, d_k)` instead of `.view(B, S, h, d_k).transpose(1, 2)` assigns wrong values to each head. The view splits $d$ into $(h, d_k)$ at the end, then transpose swaps $S$ and $h$. Reversing this silently produces incorrect results.</span>
* <span style="font-size: 14px;">**Forgetting `.contiguous()`:** After transposing back to $(B, S, h, d_k)$, memory is non-contiguous. `.view(B, S, d)` raises a runtime error. Call `.contiguous()` first, or use `.reshape()`.</span>
* <span style="font-size: 14px;">**Causal mask shape mismatch:** The mask should be $(S, S)$ or $(1, 1, S, S)$, broadcast across the batch and head dimensions. A common mistake is creating a mask of shape $(B, S, S)$ which fails to broadcast over the head dimension, or creating separate masks per head when a single shared mask suffices.</span>
* <span style="font-size: 14px;">**Dimension mismatch after concat:** After concatenating heads, the result must have exactly $d = h \times d_k$ as the last dimension. If $d$ is not divisible by $h$, the initial reshape fails. Always verify $d \mod h = 0$ before splitting.</span>
* <span style="font-size: 14px;">**Forgetting the output projection:** Omitting $W_o$ means each output dimension is determined by exactly one head with no cross-head mixing. The model loses the ability to combine information from different heads, significantly reducing capacity.</span>
* <span style="font-size: 14px;">**Applying softmax before the mask:** If you softmax first then zero out future positions, valid weights no longer sum to 1. The mask must be added before softmax.</span>
# <span style="font-size: 20px;">Sliding Window Attention</span>

<span style="font-size: 14px;">Sliding Window Attention is a sparse attention mechanism in which each token attends only to the most recent $W$ tokens (including itself) rather than every preceding token in the sequence. It was popularized by Longformer (Beltagy et al., 2020) and has since become a core building block in architectures like Mistral 7B and Gemma 3, where it is interleaved with global attention layers to balance local precision with long-range information flow.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">In standard causal self-attention, each token at position $i$ computes attention weights over every position $j \leq i$. This produces an $n \times n$ lower-triangular mask and costs $O(n^2)$ in both compute and memory. For long sequences, this quadratic cost is the dominant bottleneck.</span>

<span style="font-size: 14px;">Sliding Window Attention replaces the full causal mask with a **banded** causal mask. Token $i$ attends only to positions in $[\max(0, i - W + 1),\; i]$, where $W$ is the window size. All positions outside this band receive a mask value of $-\infty$, zeroing them out after softmax. Each token's attention is restricted to a fixed-size local neighborhood, regardless of overall sequence length.</span>

<span style="font-size: 14px;">The mechanism is **purely a masking operation**. The query, key, and value projections are unchanged. The scaled dot-product formula is unchanged. You swap the mask, and everything else stays the same, making it a drop-in modification.</span>

<span style="font-size: 14px;">In Gemma 3 (Google DeepMind, 2025), the authors set $W = 1024$ for local sliding window layers and alternate them with global attention layers that use the full causal mask. This hybrid design handles sequences up to 128K tokens without paying the quadratic cost at every layer.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">For a sequence of length $n$ and window size $W$, the mask matrix $M \in \mathbb{R}^{n \times n}$ is:</span>

$$
M[i, j] = \begin{cases} 0 & \text{if } \max(0,\; i - W + 1) \leq j \leq i \\ -\infty & \text{otherwise} \end{cases}
$$

<span style="font-size: 14px;">Here $i$ is the query position (row) and $j$ is the key position (column). A value of $0$ allows the score through; $-\infty$ kills it before softmax.</span>

<span style="font-size: 14px;">The masked attention computation is then:</span>

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}} + M\right) V
$$

<span style="font-size: 14px;">where $Q, K, V \in \mathbb{R}^{n \times d_k}$ are the query, key, and value matrices, and $d_k$ is the head dimension. Adding $M$ to the raw scores before softmax ensures that masked positions receive attention weight zero (since $e^{-\infty} = 0$).</span>

<span style="font-size: 14px;">**Complexity.** Full causal attention has $O(n^2)$ total computations (each of $n$ tokens attends to $n/2$ positions on average). Sliding window attention has $O(n \cdot W)$ since each token attends to at most $W$ positions. With $W$ fixed (e.g., 1024), this is **linear** in $n$.</span>

<span style="font-size: 14px;">Concretely, for a sequence of length $n = 8192$ and window $W = 1024$:</span>

* <span style="font-size: 14px;">**Full causal attention** computes $\frac{n(n+1)}{2} \approx 33.6\text{M}$ active entries.</span>
* <span style="font-size: 14px;">**Sliding window attention** computes roughly $n \times W \approx 8.4\text{M}$ active entries.</span>

<span style="font-size: 14px;">A 4x reduction at this length, and the gap widens as $n$ grows since full attention scales quadratically while sliding window scales linearly.</span>

---

## <span style="font-size: 16px;">Why Sliding Window Attention Works</span>

<span style="font-size: 14px;">The core linguistic observation is that **most of the information a token needs is local**. The meaning of a word depends heavily on its immediate neighbors: the same clause, the preceding sentence, the local syntactic structure. Empirically, attention weight distributions in trained full-attention models show that the vast majority of attention mass falls on nearby tokens.</span>

<span style="font-size: 14px;">Sliding window attention formalizes this as a hard constraint. By forcing each token to attend only to its $W$ nearest predecessors, per-layer computation focuses on the local context that matters most. Long-range information lost at any single layer can be recovered through the **depth of the network**, as explained in the next section.</span>

<span style="font-size: 14px;">A second motivation is **KV-cache efficiency during inference**. With full causal attention, the KV cache grows without bound. With sliding window attention, you only keep the most recent $W$ key-value pairs, capping the per-layer cache at $O(W \cdot d_k)$ regardless of generated length. This is critical for serving long-context models in production.</span>

---

## <span style="font-size: 16px;">Window Size Tradeoff</span>

<span style="font-size: 14px;">The choice of window size $W$ represents a fundamental tradeoff between **expressiveness** and **efficiency**:</span>

* <span style="font-size: 14px;">**Larger $W$:** Each token sees more context per layer, capturing longer-range dependencies directly. But the compute cost per layer increases (approaching $O(n^2)$ as $W \to n$), and the KV cache grows larger.</span>
* <span style="font-size: 14px;">**Smaller $W$:** Faster computation, smaller KV cache, and tighter focus on relevant local context. But the model becomes more dependent on network depth to propagate long-range information, and very small windows risk missing important dependencies.</span>

<span style="font-size: 14px;">In practice, model designers select $W$ based on the typical locality radius of the target domain and the compute budget. Common choices include:</span>

* <span style="font-size: 14px;">**Mistral 7B:** $W = 4096$, relatively large, with sliding window at every layer.</span>
* <span style="font-size: 14px;">**Gemma 3:** $W = 1024$, smaller, but compensated by interleaving with global attention layers every sixth layer.</span>

<span style="font-size: 14px;">The key insight is that window size should not be chosen in isolation. It must be considered alongside the total number of layers, the ratio of local to global layers, and the maximum sequence length the model will encounter.</span>

---

## <span style="font-size: 16px;">How Information Flows Beyond the Window</span>

<span style="font-size: 14px;">A natural concern with sliding window attention is: if each token can only see $W$ neighbors, how does information from the beginning of a long sequence ever reach a token at the end? The answer lies in the combination of **residual connections** and **multiple layers**.</span>

<span style="font-size: 14px;">Consider a model with $L$ sliding window layers and window size $W$. At layer 1, token $i$ attends to $[i - W + 1, i]$. At layer 2, token $i$ again attends to $[i - W + 1, i]$, but each of those positions already carries information from their own $W$-token windows at layer 1. Through the residual connection, position $i - W + 1$ at the input to layer 2 contains information from as far back as $i - 2W + 2$. This chaining continues through all $L$ layers.</span>

<span style="font-size: 14px;">The **effective receptive field** after $L$ layers is approximately $L \times W$ tokens. For Gemma 3 with $W = 1024$ and 46 layers (27B variant), the theoretical receptive field is $46 \times 1024 = 47{,}104$ tokens. Since Gemma 3 also interleaves global attention layers, information can flow even more efficiently in practice.</span>

<span style="font-size: 14px;">This is analogous to how CNNs build large receptive fields by stacking small convolutions. Sliding window attention is the sequence-modeling version of this principle.</span>

<span style="font-size: 14px;">However, the **practical** receptive field may be smaller than the theoretical one. Information degrades through many layers, and attention weights may not uniformly propagate signals from window edges. This is why hybrid architectures (mixing local and global layers) outperform pure sliding-window models on long-range reasoning tasks.</span>

---

## <span style="font-size: 16px;">Paper Context: Gemma 3</span>

<span style="font-size: 14px;">Gemma 3 (Google DeepMind, 2025) is a family of open-weight language models at 1B, 4B, 12B, and 27B parameter scales. The architecture is decoder-only and makes several notable design choices regarding attention.</span>

<span style="font-size: 14px;">The most relevant choice is the **hybrid local-global attention pattern**. The majority of layers use sliding window attention with $W = 1024$, while every sixth layer uses full global causal attention (attending to all previous positions). For the 27B model with 46 transformer layers, roughly 38 layers use local attention and 8 layers use global attention.</span>

<span style="font-size: 14px;">The motivation is twofold. First, sliding window layers are cheaper: for a 128K-token sequence, each local layer processes $O(128\text{K} \times 1024)$ entries instead of $O(128\text{K}^2)$. Second, the periodic global layers act as information highways, preventing the degradation that would occur in a pure sliding-window stack. The technical report shows this hybrid approach matches or exceeds full-attention baselines while being substantially more efficient.</span>

<span style="font-size: 14px;">Gemma 3 also uses **logit soft-capping** (capping pre-softmax logits) and **RoPE positional embeddings**. RoPE provides relative positional information that aligns naturally with the local window structure, and soft-capping stabilizes attention distributions within the constrained window.</span>

---

## <span style="font-size: 16px;">Mask Construction: Step by Step</span>

<span style="font-size: 14px;">Building the sliding window attention mask is straightforward. Given sequence length $n$ and window size $W$:</span>

<span style="font-size: 14px;">1. **Initialize** an $n \times n$ matrix $M$ filled with $-\infty$ (block everything by default).</span>

<span style="font-size: 14px;">2. **For each query position** $i$ from $0$ to $n - 1$:</span>

<span style="font-size: 14px;">3. **Compute the left boundary:** $\text{left} = \max(0, i - W + 1)$. The $\max$ with $0$ ensures we do not index before the start of the sequence.</span>

<span style="font-size: 14px;">4. **Set** $M[i, j] = 0$ for all $j$ in $[\text{left}, i]$. Token $i$ can attend to itself and the $W - 1$ tokens immediately before it (or fewer if $i < W - 1$).</span>

<span style="font-size: 14px;">5. **All other entries remain** $-\infty$, including positions $j > i$ (causal constraint) and positions $j < \text{left}$ (outside the window).</span>

<span style="font-size: 14px;">The result is a **banded lower-triangular matrix** with band width $W$. An equivalent formulation using the diagonal distance is:</span>

$$
M[i, j] = \begin{cases} 0 & \text{if } 0 \leq i - j \leq W - 1 \\ -\infty & \text{otherwise} \end{cases}
$$

<span style="font-size: 14px;">This is often more convenient for vectorized implementation, since $i - j$ is simply the signed distance from the diagonal.</span>

---

## <span style="font-size: 16px;">Numerical Example ($S = 6$, $W = 3$)</span>

<span style="font-size: 14px;">Let us construct the full $6 \times 6$ mask for a sequence of length $S = 6$ with window size $W = 3$. We denote $0$ (allow) entries as **0** and $-\infty$ (block) entries as **-inf**.</span>

<span style="font-size: 14px;">**Row $i = 0$:** left $= \max(0, 0 - 3 + 1) = 0$. Attend to $[0, 0]$.</span>

$$
[\;0,\; \text{-inf},\; \text{-inf},\; \text{-inf},\; \text{-inf},\; \text{-inf}\;]
$$

<span style="font-size: 14px;">**Row $i = 1$:** left $= \max(0, 1 - 3 + 1) = 0$. Attend to $[0, 1]$.</span>

$$
[\;0,\; 0,\; \text{-inf},\; \text{-inf},\; \text{-inf},\; \text{-inf}\;]
$$

<span style="font-size: 14px;">**Row $i = 2$:** left $= \max(0, 2 - 3 + 1) = 0$. Attend to $[0, 2]$.</span>

$$
[\;0,\; 0,\; 0,\; \text{-inf},\; \text{-inf},\; \text{-inf}\;]
$$

<span style="font-size: 14px;">**Row $i = 3$:** left $= \max(0, 3 - 3 + 1) = 1$. Attend to $[1, 3]$.</span>

$$
[\;\text{-inf},\; 0,\; 0,\; 0,\; \text{-inf},\; \text{-inf}\;]
$$

<span style="font-size: 14px;">**Row $i = 4$:** left $= \max(0, 4 - 3 + 1) = 2$. Attend to $[2, 4]$.</span>

$$
[\;\text{-inf},\; \text{-inf},\; 0,\; 0,\; 0,\; \text{-inf}\;]
$$

<span style="font-size: 14px;">**Row $i = 5$:** left $= \max(0, 5 - 3 + 1) = 3$. Attend to $[3, 5]$.</span>

$$
[\;\text{-inf},\; \text{-inf},\; \text{-inf},\; 0,\; 0,\; 0\;]
$$

<span style="font-size: 14px;">Notice the pattern: the first three rows ($i = 0, 1, 2$) look identical to standard causal attention because the window is wider than the available history. Starting at row $i = 3$, the window begins to slide and old positions get masked out.</span>

<span style="font-size: 14px;">**Contrast with full causal mask.** Full causal would allow all $j \leq i$, giving $\frac{6 \times 7}{2} = 21$ active entries. The sliding window mask has 1 + 2 + 3 + 3 + 3 + 3 = **15** active entries, eliminating about 29% of computations even at this small scale.</span>

---

## <span style="font-size: 16px;">Modern Context</span>

<span style="font-size: 14px;">Sliding window attention has a rich history in the efficient attention literature:</span>

* <span style="font-size: 14px;">**Longformer (Beltagy et al., 2020):** One of the first papers to systematically study sliding window attention. Combines local windows with global attention on special tokens like [CLS], handling 4,096+ token documents efficiently.</span>
* <span style="font-size: 14px;">**BigBird (Zaheer et al., 2020):** Extended Longformer by adding random attention connections alongside the window and global tokens. Proved that its sparse pattern is a universal approximator of sequence functions.</span>
* <span style="font-size: 14px;">**Mistral 7B (Jiang et al., 2023):** Brought sliding window attention to decoder-only LMs. Uses $W = 4096$ at every layer with no global layers, relying on multi-layer propagation for long-range dependencies.</span>
* <span style="font-size: 14px;">**Gemma 3 (Google DeepMind, 2025):** Refined the hybrid approach with $W = 1024$ interleaved with global layers every sixth layer, representing the current consensus on efficient long-context attention.</span>

<span style="font-size: 14px;">The progression shows the field converging on hybrid local-global patterns as the most practical way to scale attention to long sequences.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Off-by-one in the window boundary.** The correct left boundary is $\max(0, i - W + 1)$, giving exactly $W$ positions. A frequent error is writing $\max(0, i - W)$, which gives $W + 1$ positions. Another error is $j < i$ instead of $j \leq i$, excluding self-attention.</span>
* <span style="font-size: 14px;">**Forgetting the causal constraint.** Sliding window attention in a decoder must still be causal: token $i$ cannot attend to $j > i$. The combined constraint $\max(0, i - W + 1) \leq j \leq i$ encodes both. If you build window and causal masks separately, take the element-wise minimum when combining.</span>
* <span style="font-size: 14px;">**Applying the sliding window mask to global layers.** In Gemma 3, every sixth layer uses full global causal attention. Applying the sliding window mask to these layers cripples long-range reasoning. The mask must be selected per-layer.</span>
* <span style="font-size: 14px;">**Confusing the mask convention.** Two conventions exist: additive (0 = allow, $-\infty$ = block) and multiplicative (1 = allow, 0 = block). Gemma 3 uses additive. Mixing them up makes the mask do the opposite of what you intend.</span>
* <span style="font-size: 14px;">**Assuming the window is symmetric.** In decoder models the window is one-sided (backward-looking only). In bidirectional models like Longformer it is symmetric. For Gemma 3, a symmetric window would violate causality.</span>
* <span style="font-size: 14px;">**Neglecting KV cache truncation.** Sliding window attention bounds the KV cache at $W$ entries per layer, but only if you evict old entries. Without truncation you pay full-attention memory cost with windowed expressiveness. Use a ring buffer of size $W$, discarding the oldest entry when a new token arrives.</span>

---
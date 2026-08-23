# <span style="font-size: 20px;">Causal (Masked) Self-Attention</span>

<span style="font-size: 14px;">Causal self-attention is the mechanism that makes autoregressive language models possible. It extends standard scaled dot-product attention by adding a causal mask that prevents each token from attending to any future token, enforcing a strict left-to-right information flow. In GPT-2, every attention layer uses this causal mask, making it the defining structural choice of the decoder-only architecture.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">Standard self-attention lets every token attend to every other token in the sequence. This is useful for understanding tasks (BERT), but it is fundamentally incompatible with generation. If a model can see the future while training, it will simply copy the next token from the input instead of learning to predict it.</span>

<span style="font-size: 14px;">Causal self-attention solves this by applying a **mask** to the attention scores before the softmax. The mask sets all entries where the query position $i$ is earlier than the key position $j$ (i.e., $j > i$) to $-\infty$. After softmax, those entries become exactly zero, meaning position $i$ receives no information from any position after it. Position $i$ can only attend to positions $0, 1, \ldots, i$.</span>

<span style="font-size: 14px;">The result is an attention mechanism where each token's output depends only on itself and all preceding tokens. This is what allows GPT-2 to generate text one token at a time: at each step, the model produces a probability distribution over the next token conditioned only on the tokens that have already been generated.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">**Causal mask construction:**</span>

<span style="font-size: 14px;">Given a sequence of length $T$, the causal mask $M$ is a $T \times T$ matrix defined as:</span>

$$
M_{ij} = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}
$$

<span style="font-size: 14px;">In practice, $-\infty$ is implemented as a very large negative number (e.g., $-10^{9}$ or `-float('inf')` in PyTorch). The mask is an upper-triangular matrix of $-\infty$ values with zeros on and below the diagonal.</span>

<span style="font-size: 14px;">**Masked attention formula:**</span>

<span style="font-size: 14px;">Given queries $Q$, keys $K$, and values $V$ (all of shape $T \times d_k$ for a single head), causal attention computes:</span>

$$
\text{CausalAttn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}} + M\right) V
$$

<span style="font-size: 14px;">Breaking this into steps:</span>

<span style="font-size: 14px;">1. **Raw scores:** $S = QK^T \in \mathbb{R}^{T \times T}$, where $S_{ij} = q_i^T k_j$ measures how much query at position $i$ wants to attend to key at position $j$.</span>

<span style="font-size: 14px;">2. **Scale:** $S_{\text{scaled}} = S / \sqrt{d_k}$. Scaling prevents dot products from growing large as $d_k$ increases, which would push softmax into saturated regions with near-zero gradients.</span>

<span style="font-size: 14px;">3. **Mask:** $S_{\text{masked}} = S_{\text{scaled}} + M$. Adding $-\infty$ to future positions ensures those scores become negligible after softmax.</span>

<span style="font-size: 14px;">4. **Softmax:** $W = \text{softmax}(S_{\text{masked}})$, applied row-wise. Each row $i$ of $W$ is a valid probability distribution over positions $0$ to $i$, with zero weight on positions $i+1$ through $T-1$.</span>

<span style="font-size: 14px;">5. **Weighted sum:** $O = WV$, where each output row $o_i$ is a weighted combination of value vectors $v_0, v_1, \ldots, v_i$.</span>

---

## <span style="font-size: 16px;">Why Causal Masking</span>

<span style="font-size: 14px;">Causal masking serves three distinct purposes that together make autoregressive language modeling both correct and efficient:</span>

<span style="font-size: 14px;">**1. Preventing information leakage from the future.** The training objective is next-token prediction: given tokens $x_1, \ldots, x_t$, predict $x_{t+1}$. If position $t$ could attend to position $t+1$, it would trivially learn to copy rather than predict. The causal mask enforces a hard constraint that position $t$ can never access information beyond $t$. This is not soft regularization; it is a structural guarantee.</span>

<span style="font-size: 14px;">**2. Enabling parallel training despite sequential generation.** At inference time, generation is inherently sequential: produce one token, append it, repeat. But at training time, the entire target sequence is known. The causal mask lets the model process all $T$ positions simultaneously in a single forward pass while ensuring each position only sees the "past." The model computes the loss for all $T$ next-token predictions in one shot, making training massively parallel. Without causal masking, you would need $T$ separate forward passes, which would be computationally devastating.</span>

<span style="font-size: 14px;">**3. Matching training and inference conditions.** At inference time, when the model generates token $t$, tokens $t+1, t+2, \ldots$ do not exist yet. Causal masking during training replicates this exact condition: position $t$ never sees anything beyond itself. The model trains under the same information constraints it faces during generation, eliminating train-test mismatch.</span>

---

## <span style="font-size: 16px;">The Mask Matrix</span>

<span style="font-size: 14px;">Understanding the mask matrix in detail is essential for correct implementation. For a sequence of length $T = 4$, the causal mask $M$ looks like:</span>

$$
M = \begin{bmatrix} 0 & -\infty & -\infty & -\infty \\ 0 & 0 & -\infty & -\infty \\ 0 & 0 & 0 & -\infty \\ 0 & 0 & 0 & 0 \end{bmatrix}
$$

<span style="font-size: 14px;">**Reading the mask row by row:**</span>

* <span style="font-size: 14px;">**Row 0 (token 0):** Can only attend to itself. All other positions are masked out.</span>
* <span style="font-size: 14px;">**Row 1 (token 1):** Can attend to tokens 0 and 1. Positions 2 and 3 are masked.</span>
* <span style="font-size: 14px;">**Row 2 (token 2):** Can attend to tokens 0, 1, and 2. Position 3 is masked.</span>
* <span style="font-size: 14px;">**Row 3 (token 3):** Can attend to all tokens. Nothing is masked (last row).</span>

<span style="font-size: 14px;">**How $-\infty$ kills softmax weights:**</span>

<span style="font-size: 14px;">Softmax computes $\text{softmax}(z_j) = e^{z_j} / \sum_k e^{z_k}$. When $z_j = -\infty$, $e^{z_j} = 0$. Masked positions contribute zero to both numerator and denominator. The unmasked positions share all probability mass among themselves, producing a valid distribution over only the allowed positions.</span>

<span style="font-size: 14px;">**Implementation in code:** In PyTorch, the mask is typically constructed using `torch.triu(torch.ones(T, T), diagonal=1)` to create an upper-triangular matrix of ones (excluding the diagonal), then multiplied by a large negative value or used as a boolean mask with `masked_fill`. The key is `diagonal=1`, which selects exactly the positions where $j > i$.</span>

<span style="font-size: 14px;">**The mask is the same for every head and every layer.** In multi-head attention, the mask is broadcast across all heads. In GPT-2, every layer uses the identical causal mask, unlike encoder-decoder models where only the decoder is causal.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">GPT-2 ("Language Models are Unsupervised Multitask Learners," Radford et al., 2019) uses a **decoder-only** Transformer architecture. Unlike the original Transformer (Vaswani et al., 2017), which has both an encoder and a decoder, GPT-2 uses only the decoder stack. This is a deliberate design choice for language modeling: the goal is pure left-to-right generation, so there is no need for an encoder to process a separate input sequence.</span>

<span style="font-size: 14px;">In the original Transformer, the decoder has two attention types: **masked self-attention** (causal) and **cross-attention** (attending to encoder outputs). GPT-2 removes cross-attention entirely because there is no encoder. Every attention layer is causal self-attention. The "causal" property is not a per-layer option; it is a global architectural constraint. All 12 layers (GPT-2 Small) or 48 layers (GPT-2 XL) apply the same triangular mask.</span>

<span style="font-size: 14px;">GPT-2 also applies a pre-norm formulation (LayerNorm before attention and FFN, rather than after), but the causal mask itself is identical to what Vaswani et al. described. The innovation of GPT-2 is not the masking mechanism but the insight that a large decoder-only model trained on diverse web text can perform many tasks zero-shot.</span>

<span style="font-size: 14px;">The GPT lineage (GPT-1 through GPT-4) all use causal masking as their foundational attention pattern. Every layer preserves the causal constraint, so information can never flow backward in the sequence at any point in the network.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Let us work through a complete example with $T = 3$ tokens and $d_k = 2$.</span>

<span style="font-size: 14px;">**Queries, keys, and values (for a single head):**</span>

$$
Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}, \quad K = \begin{bmatrix} 1 & 1 \\ 0 & 1 \\ 1 & 0 \end{bmatrix}, \quad V = \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{bmatrix}
$$

<span style="font-size: 14px;">**Step 1: Compute raw scores $S = QK^T$.**</span>

$$
S = QK^T = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} 1 & 0 & 1 \\ 1 & 1 & 0 \end{bmatrix} = \begin{bmatrix} 1 & 0 & 1 \\ 1 & 1 & 0 \\ 2 & 1 & 1 \end{bmatrix}
$$

<span style="font-size: 14px;">**Step 2: Scale by $\sqrt{d_k} = \sqrt{2} \approx 1.414$.**</span>

$$
S_{\text{scaled}} = \frac{S}{\sqrt{2}} = \begin{bmatrix} 0.707 & 0 & 0.707 \\ 0.707 & 0.707 & 0 \\ 1.414 & 0.707 & 0.707 \end{bmatrix}
$$

<span style="font-size: 14px;">**Step 3: Build and add the causal mask.**</span>

$$
M = \begin{bmatrix} 0 & -\infty & -\infty \\ 0 & 0 & -\infty \\ 0 & 0 & 0 \end{bmatrix}
$$

$$
S_{\text{masked}} = S_{\text{scaled}} + M = \begin{bmatrix} 0.707 & -\infty & -\infty \\ 0.707 & 0.707 & -\infty \\ 1.414 & 0.707 & 0.707 \end{bmatrix}
$$

<span style="font-size: 14px;">**Step 4: Apply row-wise softmax.**</span>

<span style="font-size: 14px;">**Row 0:** Only one finite value ($0.707$). Softmax of a single value is $1.0$. So $W_0 = [1.0, 0, 0]$.</span>

<span style="font-size: 14px;">**Row 1:** Two finite values ($0.707, 0.707$). They are equal, so softmax gives equal weights: $W_1 = [0.5, 0.5, 0]$.</span>

<span style="font-size: 14px;">**Row 2:** Three finite values ($1.414, 0.707, 0.707$). Computing: $e^{1.414} = 4.113$, $e^{0.707} = 2.028$, $e^{0.707} = 2.028$. Sum $= 8.169$. So $W_2 = [4.113/8.169, 2.028/8.169, 2.028/8.169] = [0.503, 0.248, 0.248]$.</span>

$$
W = \begin{bmatrix} 1.0 & 0 & 0 \\ 0.5 & 0.5 & 0 \\ 0.503 & 0.248 & 0.248 \end{bmatrix}
$$

<span style="font-size: 14px;">Notice: $W$ is **lower triangular**. Every entry above the diagonal is exactly zero. This is the structural consequence of the causal mask.</span>

<span style="font-size: 14px;">**Step 5: Compute output $O = WV$.**</span>

$$
O = WV = \begin{bmatrix} 1.0 & 0 & 0 \\ 0.5 & 0.5 & 0 \\ 0.503 & 0.248 & 0.248 \end{bmatrix} \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{bmatrix}
$$

<span style="font-size: 14px;">**Row 0:** $o_0 = 1.0 \cdot [1, 2] + 0 \cdot [3, 4] + 0 \cdot [5, 6] = [1.0, 2.0]$. Token 0 sees only its own value vector.</span>

<span style="font-size: 14px;">**Row 1:** $o_1 = 0.5 \cdot [1, 2] + 0.5 \cdot [3, 4] + 0 \cdot [5, 6] = [2.0, 3.0]$. Token 1 is a 50/50 blend of $v_0$ and $v_1$.</span>

<span style="font-size: 14px;">**Row 2:** $o_2 = 0.503 \cdot [1, 2] + 0.248 \cdot [3, 4] + 0.248 \cdot [5, 6] = [2.487, 3.486]$. Token 2 attends to all three, with more weight on token 0 because its score was highest.</span>

$$
O = \begin{bmatrix} 1.0 & 2.0 \\ 2.0 & 3.0 \\ 2.487 & 3.486 \end{bmatrix}
$$

<span style="font-size: 14px;">The critical observation: row $i$ depends only on value vectors $v_0, \ldots, v_i$. No future information leaks into any position.</span>

---

## <span style="font-size: 16px;">Causal vs Bidirectional Attention</span>

<span style="font-size: 14px;">The choice between causal and bidirectional attention defines the fundamental capability of a Transformer model:</span>

* <span style="font-size: 14px;">**Causal (GPT family):** Each token attends only to previous tokens and itself. The attention weight matrix $W$ is lower triangular. This enables autoregressive generation: the model can produce text token by token, conditioning each new token on all previous ones. However, position $i$ never has access to the "full context" of the sequence because it cannot see what comes after it.</span>
* <span style="font-size: 14px;">**Bidirectional (BERT family):** Each token attends to all tokens in the sequence, both before and after. There is no mask (or equivalently, the mask is all zeros). This gives every position access to the full context, producing richer representations for understanding tasks. But it makes generation impossible because the model expects to see the complete sequence at once.</span>

<span style="font-size: 14px;">**Why not always use bidirectional?** For generation tasks (text completion, dialogue, code generation), the model must produce output sequentially. Bidirectional attention has no notion of "partial sequence" and cannot generate left to right. Causal attention is the only viable option for autoregressive generation.</span>

<span style="font-size: 14px;">**Why not always use causal?** For understanding tasks (classification, NER, question answering), restricting position $i$ to only see positions before it throws away useful information. The word "bank" at position 5 might be disambiguated by "river" at position 8, which a causal model at position 5 would never see. Bidirectional models produce better representations because every position incorporates full-sequence context.</span>

<span style="font-size: 14px;">**Hybrid approaches:** T5 and BART use bidirectional attention in the encoder and causal attention in the decoder. Prefix-LM architectures (like PaLM) use bidirectional attention for a prefix portion and causal attention for the rest.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Wrong mask direction.** The most dangerous bug: masking the lower triangle instead of the upper triangle. This creates an "anti-causal" mask where each token sees only future tokens. The model will train but produce nonsensical outputs. Always verify: `torch.triu(ones, diagonal=1)` produces the upper triangle (set to $-\infty$), meaning positions $j > i$ are masked and $j \leq i$ are visible.</span>
* <span style="font-size: 14px;">**Forgetting to add the mask before softmax.** If you apply softmax first and then zero out future positions, the remaining weights no longer sum to 1. The output becomes a scaled-down version of what it should be. The mask must be added after scaling and before softmax, so softmax normalizes only over allowed positions.</span>
* <span style="font-size: 14px;">**Mask shape mismatch with batched multi-head attention.** The score tensor has shape $(B, H, T, T)$. The mask has shape $(T, T)$ or $(1, 1, T, T)$. If you construct a mask with the wrong number of dimensions, PyTorch may silently broadcast in unexpected ways, masking wrong positions or not masking at all.</span>
* <span style="font-size: 14px;">**Using 0 instead of $-\infty$ for masked positions.** Setting masked scores to 0 does not prevent attention: $e^0 = 1$, a valid softmax weight. Masked positions still receive attention. You must use a large negative number so that $e^{-\infty} \approx 0$.</span>
* <span style="font-size: 14px;">**Off-by-one on the diagonal.** The diagonal ($i = j$, a token attending to itself) must be unmasked. Using `diagonal=0` in `torch.triu` instead of `diagonal=1` masks the diagonal too, preventing self-attention and degrading performance.</span>
* <span style="font-size: 14px;">**Applying the mask to Q/K/V instead of scores.** The mask must be added to the scaled scores $QK^T / \sqrt{d_k}$, not to the Q, K, or V matrices themselves. Confusing where in the pipeline the mask enters leads to incorrect attention patterns.</span>
* <span style="font-size: 14px;">**Assuming the mask changes per layer.** In GPT-2, every layer uses the identical causal mask. The mask depends only on sequence length $T$ and is reused across all layers and all attention heads.</span>

---
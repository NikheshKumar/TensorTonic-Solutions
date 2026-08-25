# <span style="font-size: 20px;">GPT-2 Decoder Block</span>

<span style="font-size: 14px;">The GPT-2 decoder block is the fundamental repeating unit of GPT-2. It is a pre-norm Transformer block with two sub-layers -- causal multi-head attention and a position-wise feed-forward network with GELU activation -- each preceded by LayerNorm and followed by a residual connection. GPT-2 stacks 12, 24, 36, or 48 of these identical blocks to build its four model sizes.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">A GPT-2 decoder block takes a hidden state tensor of shape $(B, T, d)$ and produces an output of the same shape. It contains exactly two sub-layers:</span>

* <span style="font-size: 14px;">**Sub-layer 1 -- Causal Multi-Head Attention:** LayerNorm normalizes the input, then causal MHA computes inter-token dependencies using a triangular mask that prevents attending to future positions. The attention output is added to the un-normalized input via a residual connection.</span>
* <span style="font-size: 14px;">**Sub-layer 2 -- Feed-Forward Network:** A second LayerNorm normalizes the result of sub-layer 1, then a two-layer FFN ($d \to 4d$ with GELU, then $4d \to d$) transforms each position independently. The FFN output is added back via a second residual connection.</span>

<span style="font-size: 14px;">The block's output has the same shape as its input, so blocks can be stacked sequentially. Each block has its own learned parameters: two LayerNorm pairs ($\gamma_1, \beta_1$ and $\gamma_2, \beta_2$), attention weights ($W_q, W_k, W_v, W_o$ with biases), and FFN weights ($W_1, b_1, W_2, b_2$). No parameters are shared between blocks.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">Let $x \in \mathbb{R}^{B \times T \times d}$ be the input to the block.</span>

### <span style="font-size: 14px;">LayerNorm</span>

<span style="font-size: 14px;">Both sub-layers begin with LayerNorm. For a single vector $z \in \mathbb{R}^d$:</span>

$$
\text{LayerNorm}(z, \gamma, \beta) = \gamma \odot \frac{z - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

<span style="font-size: 14px;">where $\mu = \frac{1}{d}\sum_{i=1}^{d} z_i$, $\sigma^2 = \frac{1}{d}\sum_{i=1}^{d}(z_i - \mu)^2$, $\gamma \in \mathbb{R}^d$ is the learned scale, $\beta \in \mathbb{R}^d$ is the learned shift, and $\epsilon = 10^{-5}$.</span>

### <span style="font-size: 14px;">Sub-layer 1: Pre-norm attention with residual</span>

$$
\hat{x} = \text{LayerNorm}(x, \gamma_1, \beta_1)
$$

$$
Q = \hat{x}W_q + b_q, \quad K = \hat{x}W_k + b_k, \quad V = \hat{x}W_v + b_v
$$

<span style="font-size: 14px;">Q, K, V are split into $h$ heads, each with dimension $d_k = d / h$. For each head:</span>

$$
\text{head}_i = \text{softmax}\!\left(\frac{Q_i K_i^T}{\sqrt{d_k}} + M\right) V_i
$$

<span style="font-size: 14px;">where $M$ is the causal mask ($M_{jk} = 0$ if $j \geq k$, $-\infty$ otherwise). Heads are concatenated and projected:</span>

$$
\text{Attn}(\hat{x}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \, W_o + b_o
$$

$$
x' = x + \text{Attn}(\hat{x})
$$

<span style="font-size: 14px;">The residual adds the attention output to the original un-normalized $x$, not to $\hat{x}$.</span>

### <span style="font-size: 14px;">Sub-layer 2: Pre-norm FFN with residual</span>

$$
\hat{x'} = \text{LayerNorm}(x', \gamma_2, \beta_2)
$$

$$
\text{FFN}(\hat{x'}) = \text{GELU}(\hat{x'} W_1 + b_1)\, W_2 + b_2
$$

$$
\text{output} = x' + \text{FFN}(\hat{x'})
$$

<span style="font-size: 14px;">where $W_1 \in \mathbb{R}^{d \times 4d}$, $b_1 \in \mathbb{R}^{4d}$, $W_2 \in \mathbb{R}^{4d \times d}$, $b_2 \in \mathbb{R}^{d}$.</span>

### <span style="font-size: 14px;">GELU activation</span>

$$
\text{GELU}(x) = x \cdot 0.5 \cdot \bigl(1 + \text{erf}(x / \sqrt{2})\bigr)
$$

<span style="font-size: 14px;">GELU is a smooth, non-monotonic activation based on the Gaussian CDF. Unlike ReLU which hard-zeros negative values, GELU softly gates them: small negative inputs are slightly suppressed while large negative inputs are nearly zeroed.</span>

---

## <span style="font-size: 16px;">Pre-Norm Architecture</span>

<span style="font-size: 14px;">GPT-2's defining structural choice is **pre-norm**: LayerNorm is applied before each sub-layer, not after. This contrasts with the original Transformer (Vaswani et al. 2017), which uses **post-norm** -- normalizing after the residual addition.</span>

<span style="font-size: 14px;">**Post-norm (original Transformer):**</span>

$$
x' = \text{LayerNorm}(x + \text{SubLayer}(x))
$$

<span style="font-size: 14px;">**Pre-norm (GPT-2):**</span>

$$
x' = x + \text{SubLayer}(\text{LayerNorm}(x))
$$

<span style="font-size: 14px;">The difference is subtle but consequential:</span>

* <span style="font-size: 14px;">**Gradient flow:** In post-norm, gradients pass through LayerNorm to reach earlier layers, dampening signals in deep networks. In pre-norm, the residual provides a direct gradient pathway bypassing normalization, making training stable for 24+ layers.</span>
* <span style="font-size: 14px;">**Learning rate sensitivity:** Post-norm requires careful warmup to avoid divergence. Pre-norm is more robust, allowing simpler training recipes.</span>
* <span style="font-size: 14px;">**Output scale:** Post-norm keeps the residual stream bounded by normalizing every block's output. In pre-norm the stream can grow, so GPT-2 adds a final LayerNorm after the last block to normalize before the LM head.</span>

<span style="font-size: 14px;">Nearly all subsequent LLMs (GPT-3, LLaMA, PaLM, Mistral) adopted pre-norm, though many switched from LayerNorm to RMSNorm.</span>

---

## <span style="font-size: 16px;">The Two Sub-Layers</span>

<span style="font-size: 14px;">The two sub-layers serve complementary roles. Understanding what each does -- and does not do -- clarifies why both are needed.</span>

### <span style="font-size: 14px;">Sub-layer 1: Causal Multi-Head Attention</span>

<span style="font-size: 14px;">Attention is the only component where tokens interact. It computes weighted combinations of value vectors, where weights come from query-key dot products. The causal mask ensures position $t$ attends only to $0, 1, \ldots, t$. Multi-head attention splits Q, K, V into $h$ heads of dimension $d_k = d/h$, each learning different attention patterns. Attention is a **mixing** operation: it redistributes information across positions without nonlinearity.</span>

### <span style="font-size: 14px;">Sub-layer 2: Position-wise Feed-Forward Network</span>

<span style="font-size: 14px;">The FFN operates on each position independently, projecting $d \to 4d$, applying GELU, then projecting $4d \to d$. The same weights apply to every position -- no inter-position interaction. The FFN is the block's **computation** layer: it introduces nonlinearity and transforms token representations. Together, the two sub-layers form a complete cycle: attention gathers context, then FFN processes each enriched representation.</span>

---

## <span style="font-size: 16px;">Why Two Residual Connections</span>

<span style="font-size: 14px;">Each sub-layer has its own independent residual connection -- two separate additions, not a single skip around the entire block:</span>

$$
x' = x + \text{Attn}(\text{LayerNorm}(x))
$$

$$
\text{output} = x' + \text{FFN}(\text{LayerNorm}(x'))
$$

<span style="font-size: 14px;">Why two instead of one?</span>

* <span style="font-size: 14px;">**Gradient highways:** Each residual creates a direct additive path for gradients. With two per block and $N$ blocks, there are $2N$ gradient highways from loss to input, preventing vanishing gradients.</span>
* <span style="font-size: 14px;">**Additive refinement:** The residual stream is the Transformer's central highway. Each sub-layer reads from it (via LayerNorm) and writes back (via addition). Attention adds contextual information; FFN adds transformed features. Each sub-layer refines the representation rather than replacing it.</span>
* <span style="font-size: 14px;">**Independent contribution:** A single residual around both sub-layers would give FFN only the raw attention output. Two residuals ensure FFN receives both the original representation and the attention contribution.</span>
* <span style="font-size: 14px;">**Training stability:** Each sub-layer can independently output near-zero early in training, defaulting to identity. Adding blocks starts as adding identity functions, preserving stability.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">GPT-2 was introduced by Radford et al. in "Language Models are Unsupervised Multitask Learners" (2019). The paper showed that scaling a next-token language model to 1.5B parameters on a diverse web corpus (WebText) yielded strong zero-shot performance across NLP benchmarks. The decoder block is the architectural core, stacked at four scales:</span>

* <span style="font-size: 14px;">**GPT-2 Small:** 12 blocks, $d = 768$, 12 heads, 117M parameters</span>
* <span style="font-size: 14px;">**GPT-2 Medium:** 24 blocks, $d = 1024$, 16 heads, 345M parameters</span>
* <span style="font-size: 14px;">**GPT-2 Large:** 36 blocks, $d = 1280$, 20 heads, 774M parameters</span>
* <span style="font-size: 14px;">**GPT-2 XL:** 48 blocks, $d = 1600$, 25 heads, 1.5B parameters</span>

<span style="font-size: 14px;">In all variants, the FFN intermediate dimension is $4d$, the head dimension is $d/h = 64$, and the context length is 1024 tokens. The vocabulary size is 50,257 (BPE tokenizer).</span>

<span style="font-size: 14px;">The switch from post-norm (GPT-1) to pre-norm was GPT-2's key change. Moving LayerNorm before each sub-layer improved stability and enabled scaling to 48 blocks. This design was retained in GPT-3 and became the LLM standard.</span>

<span style="font-size: 14px;">GPT-2 also places a final LayerNorm after the last block and before the LM head, stabilizing the residual stream before projection to logits.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Trace a single input vector through one complete decoder block with $d = 4$, $h = 2$ heads, $d_k = 2$, $d_{ff} = 16$. We use sequence length $T = 3$ but focus on position 2 (the third token).</span>

### <span style="font-size: 14px;">Input</span>

<span style="font-size: 14px;">The input at position 2: $x = [1.0, -0.5, 0.8, -0.3]$.</span>

### <span style="font-size: 14px;">Step 1: LayerNorm before attention</span>

<span style="font-size: 14px;">Mean: $\mu = (1.0 - 0.5 + 0.8 - 0.3)/4 = 0.25$. Variance: $\sigma^2 = (0.5625 + 0.5625 + 0.3025 + 0.3025)/4 = 0.4325$. Standard deviation: $\sqrt{0.4325 + 10^{-5}} \approx 0.6576$.</span>

<span style="font-size: 14px;">With $\gamma_1 = [1, 1, 1, 1]$ and $\beta_1 = [0, 0, 0, 0]$:</span>

$$
\hat{x} = \frac{[0.75, \; -0.75, \; 0.55, \; -0.55]}{0.6576} \approx [1.1408, \; -1.1408, \; 0.8365, \; -0.8365]
$$

### <span style="font-size: 14px;">Step 2: Causal multi-head attention</span>

<span style="font-size: 14px;">Project $\hat{x}$ to Q, K, V, split into 2 heads. Position 2 attends to positions 0, 1, 2 (causal mask blocks future). After attention and output projection:</span>

$$
\text{Attn}(\hat{x}) = [0.18, \; -0.12, \; 0.25, \; -0.06]
$$

### <span style="font-size: 14px;">Step 3: First residual connection</span>

$$
x' = [1.0 + 0.18, \; -0.5 - 0.12, \; 0.8 + 0.25, \; -0.3 - 0.06] = [1.18, \; -0.62, \; 1.05, \; -0.36]
$$

<span style="font-size: 14px;">We add to the original $x$, not the normalized $\hat{x}$. This is the pre-norm residual pattern.</span>

### <span style="font-size: 14px;">Step 4: LayerNorm before FFN</span>

<span style="font-size: 14px;">Mean of $x'$: $\mu' = (1.18 - 0.62 + 1.05 - 0.36) / 4 = 0.3125$. Variance: $\sigma'^2 = 0.6544$. Standard deviation: $\sqrt{0.6544 + 10^{-5}} \approx 0.8090$.</span>

<span style="font-size: 14px;">With $\gamma_2 = [1, 1, 1, 1]$ and $\beta_2 = [0, 0, 0, 0]$:</span>

$$
\hat{x'} = \frac{[0.8675, \; -0.9325, \; 0.7375, \; -0.6725]}{0.8090} \approx [1.0723, \; -1.1527, \; 0.9116, \; -0.8312]
$$

### <span style="font-size: 14px;">Step 5: Feed-forward network with GELU</span>

<span style="font-size: 14px;">Project from $d = 4$ to $4d = 16$. Three intermediate activations (before GELU): $z_1 = 0.9$, $z_2 = -0.4$, $z_3 = 1.5$:</span>

<span style="font-size: 14px;">For $z = 0.9$: $\text{GELU}(0.9) = 0.9 \times 0.5 \times (1 + \text{erf}(0.6364)) = 0.9 \times 0.5 \times 1.6319 = 0.7344$.</span>

<span style="font-size: 14px;">For $z = -0.4$: $\text{GELU}(-0.4) = -0.4 \times 0.5 \times (1 + \text{erf}(-0.2828)) = -0.4 \times 0.5 \times 0.6863 = -0.1373$.</span>

<span style="font-size: 14px;">For $z = 1.5$: $\text{GELU}(1.5) = 1.5 \times 0.5 \times (1 + \text{erf}(1.0607)) = 1.5 \times 0.5 \times 1.8584 = 1.3938$.</span>

<span style="font-size: 14px;">Large positives pass through nearly unchanged, small negatives are softly gated (not hard-zeroed like ReLU). The second linear layer projects back to $d = 4$. Suppose:</span>

$$
\text{FFN}(\hat{x'}) = [0.22, \; -0.15, \; 0.30, \; 0.08]
$$

### <span style="font-size: 14px;">Step 6: Second residual connection</span>

$$
\text{output} = [1.18 + 0.22, \; -0.62 - 0.15, \; 1.05 + 0.30, \; -0.36 + 0.08] = [1.40, \; -0.77, \; 1.35, \; -0.28]
$$

<span style="font-size: 14px;">This is the block's final output for position 2. The original information from $x$ is preserved through two residual additions, enriched with contextual information from attention and transformed features from the FFN. This output becomes the input to the next decoder block.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

### <span style="font-size: 14px;">1. Wrong norm placement: post-norm instead of pre-norm</span>

<span style="font-size: 14px;">Post-norm computes $\text{LayerNorm}(x + \text{SubLayer}(x))$, while pre-norm computes $x + \text{SubLayer}(\text{LayerNorm}(x))$. These produce completely different results. GPT-2 uses pre-norm. Applying post-norm changes the gradient flow, output magnitudes, and training dynamics. The numerical output will be wrong even if every other component is correct.</span>

### <span style="font-size: 14px;">2. Wrong residual connection targets</span>

<span style="font-size: 14px;">The residual must add the sub-layer output to the **un-normalized** input. A common mistake is writing $x' = \hat{x} + \text{Attn}(\hat{x})$ instead of $x' = x + \text{Attn}(\hat{x})$. LayerNorm is a preprocessing step for the sub-layer only. The residual always bypasses normalization, carrying the raw input forward. The same applies to the FFN sub-layer: the residual adds to $x'$, not to $\hat{x'}$.</span>

### <span style="font-size: 14px;">3. Shared LayerNorm parameters between the two sub-layers</span>

<span style="font-size: 14px;">Each sub-layer has its own independent LayerNorm with separate $\gamma$ and $\beta$ parameters: $(\gamma_1, \beta_1)$ for attention and $(\gamma_2, \beta_2)$ for FFN. These are four distinct learned vectors. Sharing a single LayerNorm between both sub-layers produces incorrect outputs because the input distributions differ -- the first normalizes raw block input, the second normalizes the post-attention residual.</span>

### <span style="font-size: 14px;">4. Forgetting the causal mask in attention</span>

<span style="font-size: 14px;">GPT-2 is autoregressive: position $t$ cannot attend to any position $t' > t$. The upper-triangular portion of the $T \times T$ score matrix must be set to $-\infty$ before softmax. Without this mask, the model sees future tokens, breaking the autoregressive property. The causal mask must be applied in every decoder block at every layer, not just the first or last.</span>

### <span style="font-size: 14px;">5. Wrong FFN activation function</span>

<span style="font-size: 14px;">GPT-2 uses GELU, not ReLU. GELU is smooth and non-monotonic, producing small negative outputs for slightly negative inputs. Substituting ReLU changes the behavior near zero, where GELU provides a soft transition rather than a hard cutoff.</span>

### <span style="font-size: 14px;">6. Missing biases in projections</span>

<span style="font-size: 14px;">GPT-2 uses biases in all linear projections ($W_q, W_k, W_v, W_o$ and $W_1, W_2$). Later models like LLaMA remove biases, but GPT-2 requires them. Omitting biases produces outputs that will not match pretrained weights.</span>

### <span style="font-size: 14px;">7. Wrong epsilon value in LayerNorm</span>

<span style="font-size: 14px;">GPT-2 uses $\epsilon = 10^{-5}$ in LayerNorm, not $10^{-6}$ or $10^{-8}$. While $\epsilon$ has minimal impact for typical inputs, edge cases with very small variance produce numerically different results with the wrong value.</span>

---
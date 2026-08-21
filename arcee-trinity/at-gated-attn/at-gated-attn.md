# <span style="font-size: 20px;">Gated Attention</span>

<span style="font-size: 14px;">Gated Attention adds a learned sigmoid gate that modulates the attention output per dimension before the residual connection. Introduced as a core component of the Arcee Trinity architecture, it gives the model fine-grained control over which attention features pass through to the next layer, suppressing irrelevant dimensions on a per-token, per-feature basis.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">In a standard Transformer block, the projected attention output is added directly to the residual stream. Every dimension contributes equally, regardless of whether that dimension carries useful information for the current token.</span>

<span style="font-size: 14px;">Gated Attention introduces a **sigmoid gate** between the attention output projection and the residual connection. The gate has shape $B \times S \times d$ (batch, sequence length, model dimension). Each element lies in $[0, 1]$, acting as a soft switch that controls how much of each attention feature passes through. A value near 1 lets the dimension pass unchanged; a value near 0 suppresses it.</span>

<span style="font-size: 14px;">The gate is computed from the same RMSNorm-normalized input that produces Q, K, and V, using a separate learned projection $W_{\text{gate}} \in \mathbb{R}^{d \times d}$ followed by the sigmoid function. This means the gating decision is informed by the same representation that drives the attention computation.</span>

---

## <span style="font-size: 16px;">What It Does</span>

<span style="font-size: 14px;">The gate provides **per-dimension, per-token filtering** of the attention output. Rather than treating the attention result as a monolithic vector, gated attention lets the model selectively amplify or suppress individual features before they enter the residual path.</span>

* <span style="font-size: 14px;">**Feature selection:** Not all $d$ dimensions carry equally useful information for every token. The gate learns which dimensions are relevant given the current context.</span>
* <span style="font-size: 14px;">**Gradient modulation:** The multiplicative gate modulates gradient magnitude during backpropagation, creating a form of learned gradient routing across feature dimensions.</span>
* <span style="font-size: 14px;">**Residual stream protection:** By zeroing out noisy attention features, the gate prevents the residual stream from accumulating spurious information across layers, especially important in deep networks where noise compounds.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">**Step 1 - RMSNorm.** The input $x \in \mathbb{R}^{B \times S \times d}$ is normalized with learnable gain $\gamma \in \mathbb{R}^{d}$:</span>

$$
\hat{x} = \text{RMSNorm}(x, \gamma) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \odot \gamma
$$

<span style="font-size: 14px;">RMSNorm omits the mean-centering of LayerNorm, reducing computation while maintaining stable training.</span>

<span style="font-size: 14px;">**Step 2 - Linear projections.** The normalized input is projected into query, key, and value spaces:</span>

$$
Q = \hat{x} \, W_Q^T, \quad K = \hat{x} \, W_K^T, \quad V = \hat{x} \, W_V^T
$$

<span style="font-size: 14px;">where $W_Q, W_K \in \mathbb{R}^{d_k \times d}$ and $W_V \in \mathbb{R}^{d_v \times d}$.</span>

<span style="font-size: 14px;">**Step 3 - Scaled dot-product attention.**</span>

$$
\text{attn} = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}} + \text{mask}\right) V
$$

<span style="font-size: 14px;">The mask is causal (for autoregressive models) or padding. Scaling by $\sqrt{d_k}$ prevents dot products from pushing softmax into saturation regions.</span>

<span style="font-size: 14px;">**Step 4 - Output projection.** The attention output is projected back to model dimension:</span>

$$
\text{attn\_proj} = \text{attn} \, W_O^T
$$

<span style="font-size: 14px;">where $W_O \in \mathbb{R}^{d \times d_v}$. In standard attention, this would be added directly to the residual.</span>

<span style="font-size: 14px;">**Step 5 - Sigmoid gate.** A separate projection of the normalized input produces the gate:</span>

$$
\text{gate} = \sigma(\hat{x} \, W_{\text{gate}}^T)
$$

<span style="font-size: 14px;">where $W_{\text{gate}} \in \mathbb{R}^{d \times d}$ and $\sigma$ is the element-wise sigmoid. The gate has shape $B \times S \times d$, matching the attention projection exactly.</span>

<span style="font-size: 14px;">**Step 6 - Gated residual connection.** The gate modulates the attention output element-wise before residual addition:</span>

$$
\text{output} = x + \text{gate} \odot \text{attn\_proj}
$$

<span style="font-size: 14px;">The $\odot$ denotes element-wise (Hadamard) multiplication. Each dimension is independently scaled by its gate value before entering the residual stream.</span>

---

## <span style="font-size: 16px;">Why Gate the Attention Output</span>

<span style="font-size: 14px;">In ungated attention, $\text{attn} \, W_O^T$ is added to the residual in its entirety. This treats all $d$ dimensions as equally important for every token at every layer.</span>

<span style="font-size: 14px;">The gate solves this by learning to suppress noisy dimensions before they reach the residual:</span>

* <span style="font-size: 14px;">**Selective information flow:** Dimensions with useful features get gate values near 1; dimensions with noise get values near 0. This filtering is independent for every token in every layer.</span>
* <span style="font-size: 14px;">**Versus ungated attention:** In a standard Transformer, the only mechanism for suppressing attention features is $W_O$, which applies the same linear transformation to all tokens. The gate adds a token-dependent, nonlinear filter that complements $W_O$.</span>
* <span style="font-size: 14px;">**Residual stream cleanliness:** In deep stacks (30+ layers), small noise in attention output compounds. The gate provides an explicit mechanism to keep the residual stream clean.</span>

---

## <span style="font-size: 16px;">The Sigmoid Gate Mechanism</span>

<span style="font-size: 14px;">The sigmoid function $\sigma(z) = \frac{1}{1 + e^{-z}}$ maps any real number to $(0, 1)$, making each gate value a continuous, differentiable switch:</span>

* <span style="font-size: 14px;">**Gate near 0:** The attention dimension is almost completely suppressed.</span>
* <span style="font-size: 14px;">**Gate near 0.5:** The dimension passes at half strength, representing maximum uncertainty.</span>
* <span style="font-size: 14px;">**Gate near 1:** The dimension passes nearly unchanged, behaving like standard ungated attention.</span>

<span style="font-size: 14px;">The gate is computed from the **same normalized input** $\hat{x}$ that produces $Q$, $K$, and $V$. This is a deliberate design choice. Because $\hat{x}$ is RMSNorm-normalized, the gate sees a stable, well-conditioned representation. Computing from raw $x$ would make gate behavior sensitive to activation scale, which varies across layers and training steps.</span>

<span style="font-size: 14px;">The shared source creates a meaningful coupling: the same representation that determines what to attend to (through $Q$, $K$) and what to aggregate (through $V$) also determines how much of that aggregated information passes through. The gate can make contextually informed decisions coherent with the attention pattern.</span>

<span style="font-size: 14px;">$W_{\text{gate}} \in \mathbb{R}^{d \times d}$ is a **separate** learned projection, adding $d^2$ parameters per layer. For $d = 4096$, this is about 16.7M parameters per layer, roughly a 25% increase over the four existing $d \times d$ matrices ($W_Q$, $W_K$, $W_V$, $W_O$). A common initialization strategy uses small weights or adds a slight positive bias so initial gate values fall around 0.6-0.7, ensuring attention information flows early in training while leaving room to learn suppression.</span>

---

## <span style="font-size: 16px;">Paper Context: Gated Attention in Arcee Trinity</span>

<span style="font-size: 14px;">Arcee Trinity incorporates gated attention as a **structural component in every Transformer block**, not an optional add-on. Each block follows this forward pass:</span>

<span style="font-size: 14px;">1. **RMSNorm** the input</span>

<span style="font-size: 14px;">2. **Compute Q, K, V** and **gate** from the normalized input</span>

<span style="font-size: 14px;">3. **Run attention** (scaled dot-product with causal mask)</span>

<span style="font-size: 14px;">4. **Apply gate** element-wise to projected attention output</span>

<span style="font-size: 14px;">5. **Add to residual** stream, then RMSNorm + FFN with its own residual</span>

<span style="font-size: 14px;">The gate placement before the residual addition is critical. If the gate were applied after residual addition, it would modulate both the attention contribution and the identity path, potentially destroying the gradient highway that residual connections provide.</span>

<span style="font-size: 14px;">Gated attention connects to a lineage of gating mechanisms:</span>

* <span style="font-size: 14px;">**LSTM (Hochreiter and Schmidhuber, 1997):** Input, forget, and output gates control cell state information flow. The forget gate is conceptually similar: it decides how much of previous state to retain.</span>
* <span style="font-size: 14px;">**GRU (Cho et al., 2014):** Simplifies LSTM to two gates. The update gate blends old and new states, analogous to how the attention gate blends residual and attention outputs.</span>
* <span style="font-size: 14px;">**GLU (Dauphin et al., 2017):** A sigmoid gate on one half of a linear projection, element-wise multiplied with the other half. Originally for convolutional models, now widely used in Transformer FFNs.</span>
* <span style="font-size: 14px;">**SwiGLU (Shazeer, 2020):** Replaces sigmoid with Swish in GLU. Used in LLaMA, PaLM, and Gemma. SwiGLU gates the FFN; Arcee Trinity's gated attention brings the same principle to the attention sublayer.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider 2 tokens with model dimension $d = 4$, single-head attention with $d_k = d_v = 4$.</span>

<span style="font-size: 14px;">**Normalized input** (after RMSNorm):</span>

$$
\hat{x} = \begin{pmatrix} 0.5 & -0.3 & 0.8 & 0.1 \\ -0.2 & 0.6 & 0.1 & -0.7 \end{pmatrix}
$$

<span style="font-size: 14px;">**Attention projection output** (after $W_O$):</span>

$$
\text{attn\_proj} = \begin{pmatrix} 0.4 & -0.6 & 0.9 & 0.2 \\ 0.3 & 0.5 & -0.1 & 0.8 \end{pmatrix}
$$

<span style="font-size: 14px;">**Gate computation.** Suppose $\hat{x} \, W_{\text{gate}}^T$ yields:</span>

$$
\begin{pmatrix} 2.0 & -1.5 & 0.3 & 1.8 \\ -2.0 & 1.2 & 0.1 & -0.5 \end{pmatrix}
$$

<span style="font-size: 14px;">Applying sigmoid element-wise:</span>

* <span style="font-size: 14px;">Token 1: $\sigma(2.0) = 0.88$, $\sigma(-1.5) = 0.18$, $\sigma(0.3) = 0.57$, $\sigma(1.8) = 0.86$</span>
* <span style="font-size: 14px;">Token 2: $\sigma(-2.0) = 0.12$, $\sigma(1.2) = 0.77$, $\sigma(0.1) = 0.52$, $\sigma(-0.5) = 0.38$</span>

$$
\text{gate} = \begin{pmatrix} 0.88 & 0.18 & 0.57 & 0.86 \\ 0.12 & 0.77 & 0.52 & 0.38 \end{pmatrix}
$$

<span style="font-size: 14px;">**Element-wise gating:**</span>

$$
\text{gate} \odot \text{attn\_proj} = \begin{pmatrix} 0.88 \times 0.4 & 0.18 \times (-0.6) & 0.57 \times 0.9 & 0.86 \times 0.2 \\ 0.12 \times 0.3 & 0.77 \times 0.5 & 0.52 \times (-0.1) & 0.38 \times 0.8 \end{pmatrix} = \begin{pmatrix} 0.352 & -0.108 & 0.513 & 0.172 \\ 0.036 & 0.385 & -0.052 & 0.304 \end{pmatrix}
$$

<span style="font-size: 14px;">**Key observations:**</span>

* <span style="font-size: 14px;">**Token 1, Dim 2 (gate = 0.18):** Attention output was $-0.6$, suppressed to $-0.108$, an 82% reduction. The model learned this dimension is not useful for token 1.</span>
* <span style="font-size: 14px;">**Token 1, Dim 1 (gate = 0.88):** Output $0.4$ passes nearly intact as $0.352$. This dimension is valuable for token 1.</span>
* <span style="font-size: 14px;">**Token 2, Dim 1 (gate = 0.12):** Almost fully suppressed ($0.3 \to 0.036$). But Token 1 had gate $0.88$ for the same dimension. This demonstrates per-token gating: the same dimension can be important for one token and irrelevant for another.</span>

<span style="font-size: 14px;">**Residual addition** (using $x = \hat{x}$ for simplicity):</span>

$$
\text{output} = x + \text{gate} \odot \text{attn\_proj} = \begin{pmatrix} 0.852 & -0.408 & 1.313 & 0.272 \\ -0.164 & 0.985 & 0.048 & -0.396 \end{pmatrix}
$$

<span style="font-size: 14px;">Compare to ungated: $x + \text{attn\_proj}$ gives Token 1 Dim 2 = $-0.9$, but gated gives $-0.408$. The gate prevents a large negative update to the residual stream.</span>

---

## <span style="font-size: 16px;">Gating in Modern Architectures</span>

<span style="font-size: 14px;">Gating is not unique to the attention sublayer. Modern Transformers apply gating in the feed-forward network as well, and the two serve complementary roles.</span>

<span style="font-size: 14px;">**GLU and SwiGLU in the FFN.** The Gated Linear Unit (Dauphin et al., 2017) splits the FFN intermediate representation into two halves: one passes through an activation (the "gate"), and the other is multiplied element-wise with it. SwiGLU (Shazeer, 2020) uses Swish instead of sigmoid and is standard in LLaMA, PaLM, Gemma, and Mistral:</span>

$$
\text{SwiGLU}(x) = (\text{Swish}(x W_1^T)) \odot (x W_2^T)
$$

<span style="font-size: 14px;">**Attention-side analogue.** SwiGLU gates the FFN's internal computation. Gated attention gates the attention sublayer's output before the residual. Together they provide gating at both processing stages of a Transformer block: the FFN gate controls "which nonlinear features to compute," the attention gate controls "which contextual features to inject."</span>

<span style="font-size: 14px;">**Gated Linear Attention.** Recent methods like GLA apply gating within the attention computation itself, replacing softmax with a gated recurrent formulation for linear-time processing. This is distinct from Arcee Trinity, which keeps softmax attention and gates only the output.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Computing the gate from un-normalized input.** The gate must use $\hat{x} = \text{RMSNorm}(x)$, not raw $x$. Without normalization, pre-sigmoid values have scale that depends on activation magnitudes, causing gates to saturate near 0 or 1 early in training or cluster near 0.5, making the gate ineffective.</span>

* <span style="font-size: 14px;">**Wrong gate shape.** A common mistake is computing a scalar gate per token ($B \times S \times 1$) instead of per-dimension ($B \times S \times d$). The scalar gate can only uniformly scale the entire attention output, losing the per-dimension selectivity that makes gated attention effective.</span>

* <span style="font-size: 14px;">**Applying the gate after the residual addition.** If the equation becomes $\text{gate} \odot (x + \text{attn\_proj})$ instead of $x + \text{gate} \odot \text{attn\_proj}$, the gate modulates the identity path as well. Low gate values destroy the residual stream, breaking the gradient highway. The gate must be applied before residual addition.</span>

* <span style="font-size: 14px;">**Sigmoid saturation.** If $W_{\text{gate}}$ is initialized with large values, pre-sigmoid activations land far from zero, pushing gates to extremes where the gradient $\sigma(z)(1 - \sigma(z))$ is near zero. This makes $W_{\text{gate}}$ nearly untrainable. Use small initialization (Xavier/Glorot) or a slight positive bias for initial gate values around 0.6-0.7.</span>

* <span style="font-size: 14px;">**Forgetting that the gate adds parameters.** $W_{\text{gate}} \in \mathbb{R}^{d \times d}$ adds $d^2$ parameters per layer. For $d = 4096$ across 32 layers, this totals about 537M extra parameters, a 5-8% increase for a 7B model. Memory, compute, and fair comparison baselines must account for this.</span>

* <span style="font-size: 14px;">**Confusing element-wise gating with attention masking.** The attention mask (additive, inside softmax) controls which tokens attend to which. The sigmoid gate (multiplicative, after output projection) controls which feature dimensions pass to the residual. They operate at completely different stages and serve different purposes.</span>

* <span style="font-size: 14px;">**Gate death.** If gate values collapse to near-zero for many dimensions, the attention branch stops receiving gradients through those dimensions, analogous to dead ReLU neurons. Monitoring gate value distributions during training is essential to detect this failure mode early.</span>

---
# <span style="font-size: 20px;">QK-Norm</span>

<span style="font-size: 14px;">QK-Norm applies RMSNorm independently to the query (Q) and key (K) matrices inside each attention head, after projection but before computing attention scores. Used in Gemma 3 (Google DeepMind, 2025), it prevents dot-product attention logits from growing uncontrollably large in deep models. Each head maintains learnable scale vectors $\gamma_q$ and $\gamma_k$ of shape $(d_{\text{head}},)$.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">In standard multi-head attention, the input hidden state $x$ is projected into queries, keys, and values via learned weight matrices, then Q and K go directly into a dot product for attention scores. QK-Norm inserts one operation between projection and dot product: it normalizes each query and key vector using RMSNorm along the $d_{\text{head}}$ dimension.</span>

<span style="font-size: 14px;">The normalization is applied independently to Q and K -- it does not touch the value (V) matrix. Each attention head has its own pair of learnable scale parameters, so heads can learn different magnitude preferences. Every query and key vector gets a controlled root-mean-square magnitude before entering the dot product, regardless of how large the projection weights produce.</span>

<span style="font-size: 14px;">In Gemma 3, the pipeline ordering is: linear projection to get Q, K, V; RMSNorm on Q and K independently; apply Rotary Position Embeddings (RoPE) to the normalized Q and K; compute dot-product attention. Normalizing before RoPE matters because RoPE is a norm-preserving rotation, so the controlled magnitude from QK-Norm carries through to the final dot product.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">Let $x \in \mathbb{R}^{B \times S \times d_{\text{model}}}$ be the input. Compute raw projections for head $h$:</span>

$$
Q_{\text{raw}} = x \, W_q^T, \quad K_{\text{raw}} = x \, W_k^T, \quad V = x \, W_v^T
$$

<span style="font-size: 14px;">where $Q_{\text{raw}}, K_{\text{raw}}, V \in \mathbb{R}^{B \times S \times d_{\text{head}}}$. Apply RMSNorm to Q along $d_{\text{head}}$:</span>

$$
\text{RMS}(Q_{\text{raw}}) = \sqrt{\frac{1}{d_{\text{head}}} \sum_{j=1}^{d_{\text{head}}} (Q_{\text{raw}})_j^2 + \epsilon}
$$

$$
Q_{\text{norm}} = \frac{Q_{\text{raw}} \cdot \gamma_q}{\text{RMS}(Q_{\text{raw}})}
$$

<span style="font-size: 14px;">Apply RMSNorm to K with its own scale:</span>

$$
\text{RMS}(K_{\text{raw}}) = \sqrt{\frac{1}{d_{\text{head}}} \sum_{j=1}^{d_{\text{head}}} (K_{\text{raw}})_j^2 + \epsilon}
$$

$$
K_{\text{norm}} = \frac{K_{\text{raw}} \cdot \gamma_k}{\text{RMS}(K_{\text{raw}})}
$$

<span style="font-size: 14px;">where $\gamma_q, \gamma_k \in \mathbb{R}^{d_{\text{head}}}$ are learnable per-head scale vectors, $\epsilon \approx 10^{-6}$ for numerical stability, and the $\gamma$ multiplication is element-wise along $d_{\text{head}}$.</span>

<span style="font-size: 14px;">After normalization, RoPE is applied:</span>

$$
Q = \text{RoPE}(Q_{\text{norm}}), \quad K = \text{RoPE}(K_{\text{norm}})
$$

<span style="font-size: 14px;">Attention scores are computed as usual:</span>

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^T}{\sqrt{d_{\text{head}}}} + \text{mask}\right) V
$$

### <span style="font-size: 14px;">Shape tracking</span>

* <span style="font-size: 14px;">$x$: $(B, S, d_{\text{model}})$ -- input hidden states.</span>
* <span style="font-size: 14px;">$W_q, W_k$: $(d_{\text{head}}, d_{\text{model}})$ -- projection matrices for one head.</span>
* <span style="font-size: 14px;">$Q_{\text{raw}}, K_{\text{raw}}$: $(B, S, d_{\text{head}})$ -- projected queries and keys before normalization.</span>
* <span style="font-size: 14px;">$\gamma_q, \gamma_k$: $(d_{\text{head}},)$ -- per-head learnable scale vectors, broadcast over $B$ and $S$.</span>
* <span style="font-size: 14px;">$\text{RMS}(\cdot)$: $(B, S, 1)$ -- scalar per token per head, reduced over $d_{\text{head}}$.</span>
* <span style="font-size: 14px;">$Q_{\text{norm}}, K_{\text{norm}}$: $(B, S, d_{\text{head}})$ -- same shape as input, magnitude controlled.</span>
* <span style="font-size: 14px;">$Q K^T$: $(B, S, S)$ -- attention logit matrix for one head.</span>

---

## <span style="font-size: 16px;">Why Normalize Q and K</span>

<span style="font-size: 14px;">The core problem QK-Norm addresses is that the magnitude of dot-product attention logits can grow dangerously large during training, especially in deep models.</span>

<span style="font-size: 14px;">**Dot-product magnitude scales with vector norms.** The dot product $q^T k$ can be as large as $\|q\| \cdot \|k\|$. The $1/\sqrt{d_{\text{head}}}$ scaling compensates for dimensional growth at initialization, but does not account for the actual magnitudes of $q$ and $k$ as they evolve during training. If norms drift upward, attention logits grow proportionally to $\|q\| \cdot \|k\| / \sqrt{d_{\text{head}}}$, and no fixed scaling factor can prevent this.</span>

<span style="font-size: 14px;">**Softmax saturation.** When attention logits become very large, the softmax saturates: it pushes almost all probability mass onto one or two tokens. The gradient becomes vanishingly small because $\frac{\partial}{\partial z_i} \text{softmax}(z)_i = p_i(1 - p_i) \approx 0$ when $p_i \approx 1$ or $p_i \approx 0$. This makes the attention layer nearly impossible to update via gradient descent.</span>

<span style="font-size: 14px;">**Attention entropy collapse.** Entire heads can "collapse" during training, converging to always attending to a single position. Once collapsed, the head contributes nothing useful. This is self-reinforcing: large logits cause peaked attention, which produces gradients only for the dominant position, further increasing the logit gap. QK-Norm breaks this loop by capping logit magnitudes.</span>

<span style="font-size: 14px;">**Norm drift across layers.** In deep Transformer stacks, the residual stream magnitude grows layer by layer. Since Q and K are linear projections of the residual stream, their norms inherit this growth. QK-Norm decouples attention logit scale from residual stream magnitude entirely.</span>

---

## <span style="font-size: 16px;">How It Differs from Standard RMSNorm</span>

<span style="font-size: 14px;">Standard RMSNorm is applied to the hidden state before attention (pre-norm) and before the feedforward block. QK-Norm uses the same formula but differs in placement and scope.</span>

* <span style="font-size: 14px;">**Location.** Standard RMSNorm sits on the main residual stream before a sub-layer. QK-Norm sits inside the attention mechanism, between the Q/K projections and the dot product -- an intra-attention operation.</span>
* <span style="font-size: 14px;">**What gets normalized.** Standard RMSNorm normalizes the full hidden state of dimension $d_{\text{model}}$. QK-Norm normalizes individual head vectors of dimension $d_{\text{head}}$, which is much smaller (e.g., $d_{\text{head}} = 256$ versus $d_{\text{model}} = 2304$ in Gemma 3 4B). Each head is normalized independently.</span>
* <span style="font-size: 14px;">**Normalization dimension.** Standard RMSNorm reduces over $d_{\text{model}}$. QK-Norm reduces over $d_{\text{head}}$ -- the last dimension of the per-head tensor but a different semantic axis.</span>
* <span style="font-size: 14px;">**Learnable parameters.** Standard RMSNorm has $\gamma$ of shape $(d_{\text{model}},)$ shared across heads. QK-Norm has separate $\gamma_q$ and $\gamma_k$ per head, each of shape $(d_{\text{head}},)$. With $H$ heads, this adds $2 \times H \times d_{\text{head}}$ parameters per layer.</span>
* <span style="font-size: 14px;">**V is untouched.** Standard RMSNorm normalizes the input that produces Q, K, and V equally. QK-Norm normalizes only Q and K, leaving V at its natural scale, because dot-product magnitude depends on Q and K norms, not V.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">Gemma 3 (Google DeepMind, 2025) uses QK-Norm in every attention layer across all model sizes. The technical report describes it as essential for training stability at scale -- without it, attention logits can diverge during long training runs, causing loss spikes.</span>

<span style="font-size: 14px;">The technique predates Gemma 3. PaLM-2 (Google, 2023) applied similar Q and K normalization. The ViT-22B paper (Dehghani et al., 2023) studied QK-Norm for Vision Transformers, showing it was necessary to scale ViTs to 22B parameters without training collapse. Chameleon (Meta, 2024) adopted it for multimodal models.</span>

<span style="font-size: 14px;">The underlying insight: standard Transformers have no mechanism to prevent Q and K projections from producing arbitrarily large vectors during training. The $1/\sqrt{d_{\text{head}}}$ scaling compensates for dimensional growth but not actual vector magnitude growth. QK-Norm adds the missing dynamic control.</span>

<span style="font-size: 14px;">Gemma 3 also uses Grouped Query Attention (GQA), where multiple query heads share a single key-value head. QK-Norm is applied to each query head and each key head independently before grouping. The shared K head is normalized once and reused across its group of query heads.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider a single attention head with $d_{\text{head}} = 4$. After projection, we have these raw vectors for one token:</span>

$$
Q_{\text{raw}} = [2.0, \; -3.0, \; 1.0, \; 4.0]
$$

$$
K_{\text{raw}} = [1.0, \; 5.0, \; -2.0, \; 0.5]
$$

<span style="font-size: 14px;">With $\gamma_q$ and $\gamma_k$ initialized to ones:</span>

$$
\gamma_q = [1.0, \; 1.0, \; 1.0, \; 1.0], \quad \gamma_k = [1.0, \; 1.0, \; 1.0, \; 1.0]
$$

### <span style="font-size: 14px;">Step 1: Compute RMS of Q</span>

$$
\text{mean}(Q_{\text{raw}}^2) = \frac{2.0^2 + (-3.0)^2 + 1.0^2 + 4.0^2}{4} = \frac{4 + 9 + 1 + 16}{4} = \frac{30}{4} = 7.5
$$

$$
\text{RMS}(Q_{\text{raw}}) = \sqrt{7.5 + 10^{-6}} \approx 2.7386
$$

### <span style="font-size: 14px;">Step 2: Normalize Q</span>

$$
Q_{\text{norm}} = \frac{Q_{\text{raw}} \cdot \gamma_q}{\text{RMS}(Q_{\text{raw}})} = \frac{[2.0, \; -3.0, \; 1.0, \; 4.0]}{2.7386} \approx [0.7303, \; -1.0954, \; 0.3651, \; 1.4606]
$$

### <span style="font-size: 14px;">Step 3: Compute RMS of K</span>

$$
\text{mean}(K_{\text{raw}}^2) = \frac{1.0^2 + 5.0^2 + (-2.0)^2 + 0.5^2}{4} = \frac{1 + 25 + 4 + 0.25}{4} = \frac{30.25}{4} = 7.5625
$$

$$
\text{RMS}(K_{\text{raw}}) = \sqrt{7.5625 + 10^{-6}} \approx 2.7501
$$

### <span style="font-size: 14px;">Step 4: Normalize K</span>

$$
K_{\text{norm}} = \frac{K_{\text{raw}} \cdot \gamma_k}{\text{RMS}(K_{\text{raw}})} = \frac{[1.0, \; 5.0, \; -2.0, \; 0.5]}{2.7501} \approx [0.3636, \; 1.8181, \; -0.7272, \; 0.1818]
$$

### <span style="font-size: 14px;">Step 5: Compare dot products before and after normalization</span>

<span style="font-size: 14px;">**Before QK-Norm (raw dot product):**</span>

$$
Q_{\text{raw}}^T K_{\text{raw}} = (2.0)(1.0) + (-3.0)(5.0) + (1.0)(-2.0) + (4.0)(0.5) = 2 - 15 - 2 + 2 = -13.0
$$

<span style="font-size: 14px;">**After QK-Norm (normalized dot product):**</span>

$$
Q_{\text{norm}}^T K_{\text{norm}} \approx (0.7303)(0.3636) + (-1.0954)(1.8181) + (0.3651)(-0.7272) + (1.4606)(0.1818)
$$

$$
\approx 0.2655 - 1.9917 - 0.2656 + 0.2655 \approx -1.7263
$$

<span style="font-size: 14px;">The raw dot product had magnitude 13.0, while the normalized dot product has magnitude approximately 1.73. After dividing by $\sqrt{d_{\text{head}}} = \sqrt{4} = 2$, the scaled attention logit goes from $-6.5$ (raw) to $-0.86$ (normalized). The normalized version produces a much softer softmax distribution, preventing saturation and maintaining gradient flow.</span>

<span style="font-size: 14px;">Now consider what happens if projection weights grow 10x during training, so $Q_{\text{raw}}$ becomes $[20, -30, 10, 40]$. Without QK-Norm, the dot product scales by $100\times$. With QK-Norm, the RMS grows proportionally, and the normalized vectors remain identical -- the dot product stays at $-1.73$. This is the key stability property.</span>

---

## <span style="font-size: 16px;">Modern Context</span>

<span style="font-size: 14px;">QK-Norm has become a standard stabilization technique in large-scale Transformer training, reflecting a trend of adding targeted normalizations within attention to enable scaling.</span>

<span style="font-size: 14px;">**PaLM-2 and Gemini.** Google's PaLM-2 (2023) used Q and K normalization to stabilize training. Gemma 3 inherits QK-Norm from this lineage, applying it across all layers and model sizes from 1B to 27B.</span>

<span style="font-size: 14px;">**ViT-22B.** Dehghani et al. (2023) studied QK-Norm for Vision Transformers, showing that without it, ViTs beyond ~4B parameters suffered attention entropy collapse. QK-Norm was the most critical modification for reaching 22B.</span>

<span style="font-size: 14px;">**Chameleon.** Meta's Chameleon (2024) adopted QK-Norm for multimodal Transformers to handle diverse magnitude distributions from image and text tokens in the same attention layer.</span>

<span style="font-size: 14px;">**Cosine attention.** A related approach L2-normalizes Q and K so the dot product becomes cosine similarity bounded in $[-1, 1]$. QK-Norm differs by using RMSNorm with learnable per-dimension $\gamma$, giving more flexibility than pure cosine attention.</span>

<span style="font-size: 14px;">**Relationship to softcapping.** Gemma 2 used logit softcapping (tanh on attention logits) to bound them. QK-Norm addresses the same problem upstream: it prevents large logits from forming by controlling input magnitudes, rather than clipping after the fact. Gemma 3 adopted QK-Norm and removed softcapping.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

<span style="font-size: 14px;">**Normalizing along the wrong dimension.** The RMS must be computed along $d_{\text{head}}$ (the last axis of the per-head tensor). Normalizing along the sequence dimension $S$ would make statistics dependent on sequence length, destroying per-token independence. Normalizing along $B$ would couple batch examples. The correct reduction axis is strictly $d_{\text{head}}$.</span>

<span style="font-size: 14px;">**Sharing gamma between Q and K.** Using a single $\gamma$ for both removes the ability to independently scale query and key magnitudes per dimension. Q and K play asymmetric roles: queries encode "what am I looking for," keys encode "what do I contain." Separate $\gamma_q$ and $\gamma_k$ let the model emphasize different features in each.</span>

<span style="font-size: 14px;">**Applying before projection instead of after.** If applied to the hidden state before the Q/K projections, QK-Norm becomes redundant with pre-attention RMSNorm and fails to control actual Q/K magnitudes. The projection matrices can still produce arbitrarily scaled outputs. QK-Norm must come after projection.</span>

<span style="font-size: 14px;">**Forgetting per-head independence.** Q and K are often stored as $(B, H, S, d_{\text{head}})$. The RMS reduction must not span the $H$ dimension. Computing RMS across all heads jointly (reducing over $H \times d_{\text{head}}$) couples the heads. Each head must have its own RMS scalar per token.</span>

<span style="font-size: 14px;">**Interaction with RoPE ordering.** QK-Norm must be applied before RoPE, not after. Applying after RoPE would disrupt the rotational position encoding. Since RoPE is norm-preserving, normalizing before it means controlled magnitudes pass through unchanged to the dot product.</span>

<span style="font-size: 14px;">**Omitting epsilon.** Without $\epsilon$, a zero vector causes division by zero. While rare, zero Q/K vectors can occur during initialization. Always include $\epsilon$ in the RMS computation.</span>

<span style="font-size: 14px;">**Incorrect initialization of gamma.** Initialize $\gamma$ to ones, not zeros. Zero initialization forces all Q and K to zero, producing uniform attention distributions and no useful signal at the start of training.</span>

---
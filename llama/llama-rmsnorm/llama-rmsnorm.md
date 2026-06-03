# <span style="font-size: 20px;">RMSNorm</span>

<span style="font-size: 14px;">Root Mean Square Layer Normalization (RMSNorm) is a simplified normalization technique that normalizes activations by their root mean square value, skipping the mean-centering step used in standard Layer Normalization. Introduced by Zhang and Sennrich (2019) and adopted as the default normalization in LLaMA (Touvron et al., 2023), RMSNorm is now the dominant normalization layer in modern large language models.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">RMSNorm is a normalization layer that divides each activation vector by its root mean square (RMS) and then applies a learned per-feature scaling parameter $\gamma$. Unlike Layer Normalization, it does not subtract the mean before normalizing. This makes it a strictly simpler operation: normalize by magnitude only, without any centering.</span>

<span style="font-size: 14px;">The core idea is that re-centering activations around zero provides negligible benefit in practice, while the re-scaling (dividing by a measure of magnitude) is what actually stabilizes training. By dropping the mean subtraction, RMSNorm reduces computational cost and simplifies the backward pass, all while maintaining comparable or identical model quality.</span>

<span style="font-size: 14px;">Given an input vector $x \in \mathbb{R}^d$, RMSNorm computes the RMS across the feature dimension, divides $x$ by that value (plus a small epsilon), and multiplies by a learnable scale vector $\gamma \in \mathbb{R}^d$. There is no learnable bias parameter, unlike LayerNorm which has both $\gamma$ and $\beta$. RMSNorm operates per-token independently, with no dependence on batch statistics, making it well-suited to autoregressive language models.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">Given an input vector $x = (x_1, x_2, \ldots, x_d) \in \mathbb{R}^d$:</span>

<span style="font-size: 14px;">**Step 1: Mean of squares.** Compute the mean of the squared elements across the feature dimension:</span>

$$
\text{MS}(x) = \frac{1}{d} \sum_{i=1}^{d} x_i^2
$$

<span style="font-size: 14px;">**Step 2: Root mean square.** Take the square root to get the RMS value:</span>

$$
\text{RMS}(x) = \sqrt{\text{MS}(x) + \epsilon} = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}
$$

<span style="font-size: 14px;">The $\epsilon$ is added inside the square root for numerical stability, preventing division by zero. LLaMA uses $\epsilon = 10^{-6}$.</span>

<span style="font-size: 14px;">**Step 3: Normalize and scale.** Divide the input by the RMS value, then multiply element-wise by the learnable scale parameter $\gamma$:</span>

$$
\text{RMSNorm}(x)_i = \frac{x_i}{\text{RMS}(x)} \cdot \gamma_i
$$

<span style="font-size: 14px;">Or equivalently, in vector form:</span>

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}} \odot \gamma
$$

<span style="font-size: 14px;">where $\odot$ denotes element-wise multiplication and $\gamma \in \mathbb{R}^d$ is the learnable scale vector.</span>

* <span style="font-size: 14px;">**$x \in \mathbb{R}^d$:** The input vector (one token's hidden state). For LLaMA-7B, $d = 4096$.</span>
* <span style="font-size: 14px;">**$d$:** The hidden dimension (number of features). The mean of squares is computed over this dimension.</span>
* <span style="font-size: 14px;">**$\epsilon$:** A small constant for numerical stability. LLaMA uses $\epsilon = 10^{-6}$. Added inside the square root, not outside.</span>
* <span style="font-size: 14px;">**$\gamma \in \mathbb{R}^d$:** Learnable per-feature scale parameter. Initialized to all ones.</span>
* <span style="font-size: 14px;">**$\text{RMS}(x)$:** A scalar computed per token. It measures the overall magnitude of the activation vector.</span>

---

## <span style="font-size: 16px;">Why Drop the Mean</span>

<span style="font-size: 14px;">Standard Layer Normalization (Ba, Kiros, and Hinton, 2016) performs two operations: it subtracts the mean (re-centering) and divides by the standard deviation (re-scaling). Zhang and Sennrich (2019) investigated whether both operations are necessary, and their key finding was that the re-scaling invariance property is the primary reason normalization works, not the re-centering.</span>

<span style="font-size: 14px;">The success of normalization comes from enforcing scale invariance. If the weights in the preceding linear layer are scaled by $\alpha$, the activations scale by $\alpha$, but the RMS also scales by $\alpha$, so $x / \text{RMS}(x)$ is unchanged. This stabilizes gradients regardless of weight magnitude. The mean subtraction provides re-centering invariance (shift-invariance), but Zhang and Sennrich showed empirically that this property contributes minimally to training stability.</span>

<span style="font-size: 14px;">Computationally, dropping the mean eliminates one reduction operation, the subtraction across all $d$ features, and the variance computation. Instead of two passes over the feature vector (mean, then variance), RMSNorm needs one pass (mean of squares). The backward pass also simplifies: LayerNorm differentiates through both the mean and variance, while RMSNorm only differentiates through the root mean square.</span>

<span style="font-size: 14px;">Zhang and Sennrich validated this across multiple NLP tasks (machine translation, abstractive summarization, language modeling), showing that RMSNorm matches LayerNorm's performance with a 7-64% speedup on the normalization operation depending on architecture and hardware.</span>

---

## <span style="font-size: 16px;">RMSNorm vs LayerNorm</span>

<span style="font-size: 14px;">LayerNorm computes the mean and variance across the feature dimension, subtracts the mean, and divides by the standard deviation:</span>

$$
\text{LayerNorm}(x)_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma_i + \beta_i, \quad \mu = \frac{1}{d} \sum_{i=1}^{d} x_i, \quad \sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2
$$

<span style="font-size: 14px;">The differences from RMSNorm are precise:</span>

* <span style="font-size: 14px;">**No mean subtraction:** LayerNorm subtracts $\mu$ from each element before dividing. RMSNorm does not. This is the defining difference.</span>
* <span style="font-size: 14px;">**Different divisor:** LayerNorm divides by $\sigma$ (standard deviation). RMSNorm divides by $\text{RMS}(x)$. They are related: $\text{RMS}(x)^2 = \mu^2 + \sigma^2$. When $\mu = 0$, they coincide.</span>
* <span style="font-size: 14px;">**No bias parameter:** LayerNorm has both $\gamma$ (scale) and $\beta$ (shift). RMSNorm has only $\gamma$. Dropping $\beta$ is deliberate: it could re-introduce a non-zero mean, partially undoing the normalization.</span>
* <span style="font-size: 14px;">**Fewer reductions:** LayerNorm requires two reductions (mean and variance). RMSNorm requires one (mean of squares). A constant-factor improvement, but meaningful at scale.</span>
* <span style="font-size: 14px;">**Fewer parameters:** LayerNorm has $2d$ parameters ($\gamma$ and $\beta$), while RMSNorm has $d$ (only $\gamma$). For LLaMA-7B with $d = 4096$ and 64 RMSNorm layers, this saves $64 \times 4096 = 262{,}144$ parameters. Small relative to 7B, but the simplification is principled.</span>

---

## <span style="font-size: 16px;">The Gamma Parameter</span>

<span style="font-size: 14px;">The learnable scale parameter $\gamma \in \mathbb{R}^d$ is a per-feature weight that modulates each dimension of the normalized output independently. It is the only trainable parameter in RMSNorm.</span>

<span style="font-size: 14px;">$\gamma$ is initialized to all ones. At initialization, RMSNorm acts as pure normalization with no additional scaling. During training, each $\gamma_i$ is free to grow or shrink, learning the optimal scale for each feature dimension.</span>

<span style="font-size: 14px;">The role of $\gamma$ is to restore representational power that raw normalization removes. Without $\gamma$, all outputs are constrained to unit-RMS vectors. With $\gamma$, the network can amplify important features and suppress irrelevant ones. If the network learns $\gamma_i = \text{RMS}(x)$ for all $i$, it recovers the original input exactly, so RMSNorm never reduces model capacity.</span>

<span style="font-size: 14px;">In practice, $\gamma$ values after training cluster near 1.0 but with meaningful variation. In PyTorch, $\gamma$ is stored as an `nn.Parameter` of shape `(d_model,)` and receives gradients through standard backpropagation.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">The LLaMA paper (Touvron et al., 2023) adopted RMSNorm as one of several architectural improvements over the original Transformer, stating: "We normalize the input of each transformer sub-layer, instead of normalizing the output. We use the RMSNorm normalizing function, introduced by Zhang and Sennrich (2019)."</span>

<span style="font-size: 14px;">Two design decisions in LLaMA's use of RMSNorm are worth highlighting:</span>

* <span style="font-size: 14px;">**Pre-norm placement:** RMSNorm is applied before each sub-layer (self-attention and feed-forward), not after. The computation is $x + \text{Sublayer}(\text{RMSNorm}(x))$ rather than $\text{RMSNorm}(x + \text{Sublayer}(x))$. This pre-norm architecture (Xiong et al., 2020) provides more stable gradients and avoids the vanishing gradient issues that post-norm encounters in very deep networks.</span>
* <span style="font-size: 14px;">**Every sub-layer:** In each LLaMA transformer block, RMSNorm is applied twice: once before self-attention and once before the SwiGLU feed-forward sub-layer. For LLaMA-7B with 32 layers, this means 64 RMSNorm operations per forward pass, plus a final RMSNorm before the output projection.</span>

<span style="font-size: 14px;">LLaMA uses $\epsilon = 10^{-6}$, smaller than the typical LayerNorm default of $10^{-5}$. RMSNorm is generally more numerically stable since it avoids the variance computation, which can be problematic when variance is near zero.</span>

<span style="font-size: 14px;">This setup became the de facto standard. LLaMA 2, LLaMA 3, Mistral (Jiang et al., 2023), Gemma (Google DeepMind, 2024), and Qwen (Bai et al., 2023) all use pre-norm RMSNorm, validating it as a robust default for large-scale autoregressive language models.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider $d = 4$ with a concrete input vector and gamma, mirroring one token position inside a LLaMA layer with a reduced dimension for clarity.</span>

<span style="font-size: 14px;">**Input vector:**</span>

$$
x = (2.0, \; -1.0, \; 3.0, \; -4.0)
$$

<span style="font-size: 14px;">**Gamma (learned scale):**</span>

$$
\gamma = (1.0, \; 0.5, \; 1.2, \; 0.8)
$$

<span style="font-size: 14px;">**Epsilon:** $\epsilon = 10^{-6}$</span>

<span style="font-size: 14px;">**Step 1: Compute the mean of squares.**</span>

$$
\text{MS}(x) = \frac{1}{4}(2.0^2 + (-1.0)^2 + 3.0^2 + (-4.0)^2) = \frac{1}{4}(4.0 + 1.0 + 9.0 + 16.0) = \frac{30.0}{4} = 7.5
$$

<span style="font-size: 14px;">Note that signs do not matter because squaring makes everything positive. This is why RMSNorm works without mean subtraction.</span>

<span style="font-size: 14px;">**Step 2: Compute the RMS.**</span>

$$
\text{RMS}(x) = \sqrt{7.5 + 10^{-6}} \approx 2.7386
$$

<span style="font-size: 14px;">The epsilon has negligible effect here. It only matters when the input vector is near-zero.</span>

<span style="font-size: 14px;">**Step 3: Normalize (divide by RMS).**</span>

$$
\hat{x} = \frac{x}{\text{RMS}(x)} = \frac{1}{2.7386}(2.0, \; -1.0, \; 3.0, \; -4.0)
$$

$$
\hat{x} \approx (0.7303, \; -0.3651, \; 1.0954, \; -1.4606)
$$

<span style="font-size: 14px;">After normalization, the vector's RMS is approximately 1.0. Verification: $\frac{1}{4}(0.7303^2 + 0.3651^2 + 1.0954^2 + 1.4606^2) \approx \frac{1}{4}(0.5333 + 0.1333 + 1.1999 + 2.1333) = 1.0$.</span>

<span style="font-size: 14px;">**Step 4: Apply gamma (element-wise multiply).**</span>

$$
\text{RMSNorm}(x) = \hat{x} \odot \gamma = (0.7303 \times 1.0, \; -0.3651 \times 0.5, \; 1.0954 \times 1.2, \; -1.4606 \times 0.8)
$$

$$
\text{RMSNorm}(x) \approx (0.7303, \; -0.1826, \; 1.3145, \; -1.1685)
$$

<span style="font-size: 14px;">Gamma has amplified the third feature (scale 1.2), suppressed the second (scale 0.5), left the first unchanged (scale 1.0), and slightly reduced the fourth (scale 0.8). This is how the network learns per-feature importance through $\gamma$.</span>

<span style="font-size: 14px;">**Contrast with LayerNorm:** For this input, $\mu = \frac{1}{4}(2 - 1 + 3 - 4) = 0$, so LayerNorm and RMSNorm produce identical results. When $\mu \neq 0$, the outputs diverge because LayerNorm subtracts the mean first.</span>

---

## <span style="font-size: 16px;">Modern Context</span>

<span style="font-size: 14px;">RMSNorm has become the standard normalization layer in virtually all modern large language models. The original Transformer (Vaswani et al., 2017) used post-norm LayerNorm. GPT-2 (Radford et al., 2019) switched to pre-norm LayerNorm. LLaMA (Touvron et al., 2023) replaced LayerNorm with RMSNorm. Since then, nearly every open-weight LLM has followed: LLaMA 2, LLaMA 3, Mistral, Mixtral, Gemma, Qwen, Yi, DeepSeek, and others all use pre-norm RMSNorm.</span>

<span style="font-size: 14px;">Hardware vendors have optimized for this convergence. NVIDIA's cuDNN and Triton include RMSNorm kernels, and inference frameworks like vLLM and TensorRT-LLM use fused CUDA kernels. The simplicity of the operation (one reduction, one division, one element-wise multiply) makes it amenable to kernel fusion.</span>

<span style="font-size: 14px;">Encoder-only and encoder-decoder models (BERT, T5) continue to use LayerNorm, partly because these architectures predate RMSNorm's adoption. For decoder-only autoregressive models, which dominate current LLM development, RMSNorm is the clear default.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Wrong dimension for the mean of squares.** The mean must be computed over the last (feature) dimension, not the sequence or batch dimension. Computing `x.pow(2).mean(dim=0)` or `x.pow(2).mean(dim=1)` instead of `x.pow(2).mean(dim=-1)` normalizes across the wrong axis, producing silently incorrect outputs.</span>
* <span style="font-size: 14px;">**Forgetting keepdim in the mean computation.** The mean of squares must retain its dimensionality for correct broadcasting. Without `keepdim=True`, the shape collapses from `(batch, seq_len, 1)` to `(batch, seq_len)`, and the subsequent `x / rms` division broadcasts incorrectly, either raising a shape error or silently dividing wrong elements.</span>
* <span style="font-size: 14px;">**Epsilon placed outside the square root.** Writing $\sqrt{\text{MS}(x)} + \epsilon$ instead of $\sqrt{\text{MS}(x) + \epsilon}$ seems equivalent but is not. When $\text{MS}(x) = 0$, the gradient through $\sqrt{0}$ is infinite, causing NaN during backpropagation. Placing epsilon inside the square root avoids this singularity. Additionally, an epsilon that is too small (like $10^{-12}$) can cause float16 underflow in mixed-precision training.</span>
* <span style="font-size: 14px;">**Confusing RMSNorm with LayerNorm.** Starting from a LayerNorm implementation and "just removing the mean" is error-prone. LayerNorm divides by $\sigma = \sqrt{\text{Var}(x) + \epsilon}$ where $\text{Var}(x) = \mathbb{E}[x^2] - (\mathbb{E}[x])^2$. RMSNorm divides by $\sqrt{\mathbb{E}[x^2] + \epsilon}$. The variance subtracts the squared mean; the mean of squares does not. Using variance instead of mean of squares produces wrong results whenever $\mu \neq 0$.</span>
* <span style="font-size: 14px;">**Forgetting the gamma multiplication.** RMSNorm without $\gamma$ constrains all outputs to have unit RMS, reducing model capacity. If $\gamma$ is omitted or not registered as an `nn.Parameter`, the model can still train but will underperform because it cannot learn per-feature scaling.</span>
* <span style="font-size: 14px;">**Initializing gamma to zeros instead of ones.** If $\gamma$ starts at zero, all RMSNorm outputs are zero, meaning the residual connection passes through only the identity. This effectively disables the sub-layer at initialization and can cause training instability. The correct initialization is all ones.</span>
* <span style="font-size: 14px;">**Applying RMSNorm after the sub-layer instead of before.** LLaMA uses pre-norm: $x + \text{Sublayer}(\text{RMSNorm}(x))$. Post-norm placement, $\text{RMSNorm}(x + \text{Sublayer}(x))$, changes gradient flow and can cause training instability in deep networks. The placement matters more than the choice of normalization function.</span>

---
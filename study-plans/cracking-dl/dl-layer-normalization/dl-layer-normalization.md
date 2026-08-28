# <span style="font-size: 20px;">Layer Normalization</span>

<span style="font-size: 14px;">Layer Normalization (Ba et al., 2016) is the normalization technique used in virtually every Transformer model. While Batch Normalization normalizes across the batch, Layer Normalization normalizes across the features within each individual sample. This distinction makes it essential to understand for any deep learning interview.</span>

---

## <span style="font-size: 16px;">Layer Norm vs Batch Norm</span>

<span style="font-size: 14px;">The fundamental difference lies in the normalization axis:</span>

- <span style="font-size: 14px;">**Batch Norm**: for each feature $j$, compute mean and variance across the batch: $\mu_j = (1/N) \sum_{i=1}^N x_{ij}$. This makes each feature zero-mean and unit-variance across samples.</span>
- <span style="font-size: 14px;">**Layer Norm**: for each sample $i$, compute mean and variance across features: $\mu_i = (1/D) \sum_{j=1}^D x_{ij}$. This makes each sample's feature vector zero-mean and unit-variance.</span>

<span style="font-size: 14px;">Key consequences of this difference:</span>

- <span style="font-size: 14px;">**No batch dependence**: Layer Norm's statistics are computed per sample, so there is no difference between training and inference behavior. No running mean/variance is needed.</span>
- <span style="font-size: 14px;">**Works with any batch size**: including batch size 1, which is common during autoregressive generation in LLMs.</span>
- <span style="font-size: 14px;">**Sequence-friendly**: for sequences of varying length, Batch Norm would mix statistics across positions, which is semantically wrong. Layer Norm treats each token independently.</span>

---

## <span style="font-size: 16px;">Forward Pass</span>

<span style="font-size: 14px;">Given input $x \in \mathbb{R}^{N \times D}$, the forward pass computes:</span>

$$
\mu_i = \frac{1}{D} \sum_{j=1}^D x_{ij}, \quad \sigma_i^2 = \frac{1}{D} \sum_{j=1}^D (x_{ij} - \mu_i)^2
$$

$$
\begin{aligned}
\hat{x}_{ij} = \frac{x_{ij} - \mu_i}{\sqrt{\sigma_i^2 + \varepsilon}}, \\
y_{ij} = \gamma_j \hat{x}_{ij} + \beta_j
\end{aligned}
$$

<span style="font-size: 14px;">The $\varepsilon$ term (typically $10^{-5}$) prevents division by zero when all features are equal. The learnable parameters $\gamma$ (scale) and $\beta$ (shift) allow the network to undo the normalization if needed - they restore the representational power that normalization removes.</span>

---

## <span style="font-size: 16px;">Backward Pass</span>

<span style="font-size: 14px;">The backward pass computes gradients for $x$, $\gamma$, and $\beta$. Given upstream gradient $\partial L / \partial y$:</span>

<span style="font-size: 14px;">**Gradients for learnable parameters** (straightforward):</span>

$$
\begin{aligned}
\frac{\partial L}{\partial \gamma_j} &= \sum_{i=1}^N \frac{\partial L}{\partial y_{ij}} \hat{x}_{ij} \\[6pt]
\frac{\partial L}{\partial \beta_j} &= \sum_{i=1}^N \frac{\partial L}{\partial y_{ij}}
\end{aligned}
$$

<span style="font-size: 14px;">**Gradient for input** (the non-trivial part): let $g = \partial L / \partial y \odot \gamma$ (gradient at the $\hat{x}$ level). The gradient through normalization has a compact form:</span>

$$
\begin{aligned}
\frac{\partial L}{\partial x_i} = \frac{1}{\sqrt{\sigma_i^2 + \varepsilon}} \bigl(& g_i - \frac{1}{D} \sum_j g_{ij} \\
&- \frac{\hat{x}_i}{D} \sum_j g_{ij} \hat{x}_{ij} \bigr)
\end{aligned}
$$

<span style="font-size: 14px;">This formula shows that the gradient is the original upstream signal minus two correction terms: one for the mean shift and one for the variance scaling. The correction terms ensure that the gradient of normalized outputs sums to zero (mean correction) and is orthogonal to $\hat{x}$ (variance correction).</span>

---

## <span style="font-size: 16px;">Layer Norm in Transformers</span>

<span style="font-size: 14px;">In the original Transformer ("Attention Is All You Need"), Layer Norm is applied after the residual addition: $\text{LN}(x + \text{SubLayer}(x))$. This is called **Post-LN**.</span>

<span style="font-size: 14px;">Most modern architectures use **Pre-LN** instead: $x + \text{SubLayer}(\text{LN}(x))$. Pre-LN places normalization before the attention/FFN sublayer. This is preferred because:</span>

- <span style="font-size: 14px;">**Training stability**: Pre-LN keeps the residual stream unnormalized, preventing gradient explosion in deep models. GPT-2, GPT-3, LLaMA all use Pre-LN.</span>
- <span style="font-size: 14px;">**No warm-up needed**: Post-LN typically requires learning rate warm-up to avoid divergence in early training.</span>
- <span style="font-size: 14px;">**Better gradient flow**: the residual connection provides an unmodified gradient path, and LN acts as a "stabilizer" within each sublayer.</span>

<span style="font-size: 14px;">**RMSNorm** (Zhang and Sennrich, 2019) is a simplified variant that drops the mean centering: $\hat{x} = x / \text{RMS}(x)$ where $\text{RMS}(x) = \sqrt{(1/D) \sum x_j^2}$. Used in LLaMA, Mistral, and other modern LLMs for its computational simplicity (one fewer reduction operation).</span>

---


## <span style="font-size: 16px;">RMSNorm and Modern Variants</span>

<span style="font-size: 14px;">**RMSNorm** drops the mean-centering step and normalizes by the root-mean-square only: x / sqrt(mean(x^2) + eps) * gamma. This removes one reduction operation per normalization, giving roughly 10-15% speedup with negligible quality difference. LLaMA and most modern LLMs use RMSNorm.</span>

<span style="font-size: 14px;">**Pre-norm vs post-norm.** The original transformer uses post-norm (normalize after the residual add), but this requires careful learning rate warmup. Pre-norm (normalize before the sublayer) is more stable and is the default in modern architectures. However, some research suggests post-norm achieves slightly better final quality with proper training.</span>

<span style="font-size: 14px;">**Why not BatchNorm in transformers?** Batch normalization normalizes across the batch dimension, which is problematic for sequences of varying length. It also breaks the autoregressive property during generation since batch statistics would depend on what other sequences are in the batch. Layer normalization normalizes each sample independently, making it compatible with variable-length sequences and autoregressive decoding.</span>


## <span style="font-size: 16px;">Common Interview Follow-ups</span>

- <span style="font-size: 14px;">**Q: Why can't Batch Norm be used in Transformers?**</span>
  <span style="font-size: 14px;">A: Three reasons. First, during autoregressive generation, batch size is effectively 1, so batch statistics are meaningless. Second, sequences in a batch may have different lengths, so normalizing across the batch at each position mixes padded and real tokens. Third, the batch statistics during training depend on what other samples happen to be in the mini-batch, introducing unwanted coupling between samples. Layer Norm avoids all these issues by normalizing each sample independently.</span>

- <span style="font-size: 14px;">**Q: What does the backward gradient formula tell us about Layer Norm's effect on gradients?**</span>
  <span style="font-size: 14px;">A: The formula projects out two components from the gradient: the mean (making the gradient sum to zero across features) and the component along $\hat{x}$ (making the gradient orthogonal to the normalized output). This acts as an implicit regularizer - it prevents the gradient from simply scaling $\hat{x}$ up, forcing the network to learn more diverse feature updates.</span>

- <span style="font-size: 14px;">**Q: Why do LLaMA and Mistral use RMSNorm instead of Layer Norm?**</span>
  <span style="font-size: 14px;">A: RMSNorm removes the mean subtraction step, computing only $x / \sqrt{(1/D)\sum x_j^2 + \varepsilon}$. This is computationally cheaper (one fewer reduction over the feature dimension) and empirically performs similarly. The intuition is that re-centering is less important than re-scaling for Transformer training stability.</span>

- <span style="font-size: 14px;">**Q: Pre-LN vs Post-LN - which is better and why?**</span>
  <span style="font-size: 14px;">A: Pre-LN is more stable and used in most modern models (GPT-2/3, LLaMA). Post-LN can achieve slightly better final performance but requires careful learning rate warm-up and often diverges without it. The key insight is that Pre-LN preserves the residual stream magnitude, while Post-LN normalizes it away, potentially losing information about scale.</span>

- <span style="font-size: 14px;">**Q: What is the variance of the biased estimator used in Layer Norm?**</span>
  <span style="font-size: 14px;">A: Layer Norm uses the biased variance $(1/D) \sum (x - \mu)^2$ rather than the unbiased Bessel-corrected $(1/(D-1))$ version. This is standard in normalization layers because: (a) $D$ is typically large (768+) so the difference is negligible, (b) the learnable $\gamma$ can compensate for any scaling difference, and (c) the biased version is slightly cheaper to compute.</span>

---
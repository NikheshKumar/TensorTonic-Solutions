# <span style="font-size: 20px;">Layer Normalization</span>

<span style="font-size: 14px;">Layer Normalization (LayerNorm) is a technique that normalizes activations across the feature dimension for each individual token, stabilizing training and enabling faster convergence. Introduced by Ba, Kiros, and Hinton (2016), it became a critical component of Transformer architectures. GPT-2 specifically uses a "pre-norm" variant, applying LayerNorm before each sub-layer rather than after.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">Layer Normalization normalizes activations within each individual sample across the feature dimension. Given a hidden vector $x \in \mathbb{R}^d$ for a single token, LayerNorm computes the mean and variance over all $d$ elements, subtracts the mean, divides by the standard deviation, then applies a learned affine transformation. The output has zero mean and unit variance before the affine step, providing a stable distribution for the next layer to consume.</span>

<span style="font-size: 14px;">Unlike Batch Normalization, which computes statistics across the batch dimension, LayerNorm operates entirely within a single example. This makes it independent of batch size and naturally suited to sequence models. Every token is normalized using only its own feature values, with no dependence on other tokens in the batch or other positions in the sequence.</span>

<span style="font-size: 14px;">In GPT-2, LayerNorm appears twice per Transformer block: once before the multi-head self-attention sub-layer, and once before the feed-forward network sub-layer. A final LayerNorm is applied after the last block before the output projection. The learnable parameters $\gamma$ (scale) and $\beta$ (shift) each have dimension $d_{\text{model}}$, meaning GPT-2 Small (768-dimensional hidden states) has 768 learnable parameters per LayerNorm instance.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">Given an input vector $x = [x_1, x_2, \dots, x_d]$ of dimension $d$, LayerNorm proceeds in four steps.</span>

<span style="font-size: 14px;">**Step 1 -- Compute the mean.** Average all elements of $x$:</span>

$$
\mu = \frac{1}{d} \sum_{i=1}^{d} x_i
$$

<span style="font-size: 14px;">This is a scalar computed from the single token's hidden vector.</span>

<span style="font-size: 14px;">**Step 2 -- Compute the variance.** Measure the spread of elements around the mean:</span>

$$
\sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2
$$

<span style="font-size: 14px;">This uses population variance (dividing by $d$, not $d - 1$), which is standard in LayerNorm implementations.</span>

<span style="font-size: 14px;">**Step 3 -- Normalize.** Center and scale each element:</span>

$$
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

<span style="font-size: 14px;">The small constant $\epsilon$ (typically $10^{-5}$ in GPT-2) prevents division by zero when the variance is extremely small. After this step, $\hat{x}$ has approximately zero mean and unit variance.</span>

<span style="font-size: 14px;">**Step 4 -- Scale and shift.** Apply learned affine parameters:</span>

$$
\text{LayerNorm}(x)_i = \gamma_i \cdot \hat{x}_i + \beta_i
$$

<span style="font-size: 14px;">Here $\gamma \in \mathbb{R}^d$ and $\beta \in \mathbb{R}^d$ are learnable parameter vectors, initialized to $\gamma = \mathbf{1}$ and $\beta = \mathbf{0}$ so that LayerNorm acts as identity-after-normalization at initialization. During training, the network learns to scale and shift each dimension as needed.</span>

<span style="font-size: 14px;">Putting it all together in a single expression:</span>

$$
\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

<span style="font-size: 14px;">where $\odot$ denotes element-wise multiplication. The entire operation is differentiable, so gradients flow through the normalization during backpropagation.</span>

---

## <span style="font-size: 16px;">Pre-Norm vs Post-Norm</span>

<span style="font-size: 14px;">The original Transformer (Vaswani et al., 2017) uses **post-norm** placement: the residual connection is added first, then LayerNorm is applied. GPT-2 switches to **pre-norm** placement: LayerNorm is applied before the sub-layer, and the residual connection bypasses the entire normalized sub-layer.</span>

<span style="font-size: 14px;">**Post-Norm (Original Transformer):**</span>

$$
h = \text{LayerNorm}(x + \text{SubLayer}(x))
$$

<span style="font-size: 14px;">In post-norm, the residual path passes through LayerNorm at every block. For deep networks, this creates a chain of normalization operations on the residual stream, which can cause gradient magnitude issues. Post-norm often requires careful learning rate warmup to avoid divergence.</span>

<span style="font-size: 14px;">**Pre-Norm (GPT-2):**</span>

$$
h = x + \text{SubLayer}(\text{LayerNorm}(x))
$$

<span style="font-size: 14px;">In pre-norm, the un-normalized original $x$ is added back via the residual connection, creating a clean gradient highway from output to input embedding. With post-norm, backpropagation encounters LayerNorm's Jacobian at every layer, which can compound and cause instability. With pre-norm, the gradient includes a direct identity term, ensuring gradients neither vanish nor explode regardless of depth.</span>

<span style="font-size: 14px;">**Practical differences:**</span>

* <span style="font-size: 14px;">**Post-norm** typically requires learning rate warmup (the original Transformer uses 4,000 warmup steps).</span>
* <span style="font-size: 14px;">**Pre-norm** is more forgiving and can often train without warmup or with shorter warmup periods.</span>
* <span style="font-size: 14px;">**Pre-norm** is the dominant choice in large language models: GPT-2, GPT-3, PaLM, LLaMA, and most modern architectures use it.</span>

<span style="font-size: 14px;">GPT-2 also adds an extra LayerNorm after the final Transformer block, before the language model head, ensuring the representations are well-conditioned before the output projection.</span>

---

## <span style="font-size: 16px;">Why Normalize at All</span>

<span style="font-size: 14px;">**Internal covariate shift.** Ioffe and Szegedy (2015) coined this term to describe how the distribution of inputs to each layer shifts as preceding layers update their parameters. Each layer must continuously re-adapt to new input statistics, slowing convergence. While the precise mechanism is debated (Santurkar et al., 2018 argue normalization works primarily by smoothing the loss landscape), the practical benefits are clear: normalized networks train faster and are less sensitive to initialization and learning rate choices.</span>

<span style="font-size: 14px;">**Gradient flow.** Without normalization, activations can grow or shrink exponentially with depth. LayerNorm keeps activations in a bounded range at every layer, ensuring gradients maintain a healthy magnitude throughout the network. This is especially critical for Transformers, which are often 12, 24, 48, or even 96 layers deep.</span>

<span style="font-size: 14px;">**Decoupling magnitude from direction.** After normalization, the activation vector has unit variance. The learned $\gamma$ and $\beta$ parameters then control the magnitude and bias separately from the direction of the activation vector. This decomposition gives the optimizer independent control over scale and orientation, simplifying the optimization landscape.</span>

---

## <span style="font-size: 16px;">LayerNorm vs BatchNorm</span>

<span style="font-size: 14px;">Batch Normalization (BatchNorm, Ioffe and Szegedy, 2015) normalizes across the batch dimension: for each feature, it computes statistics over all examples in the mini-batch. LayerNorm normalizes across the feature dimension: for each example, it computes statistics over all features.</span>

<span style="font-size: 14px;">**Why BatchNorm fails for sequences.** In language modeling, sequences have variable length, and the "meaning" of position $t$ varies across examples in a batch. Computing per-feature statistics across positions conflates different semantic roles. At generation time, the model processes one token at a time with a batch size of 1, making batch statistics undefined. LayerNorm avoids all these problems by operating on each token independently.</span>

<span style="font-size: 14px;">**Key differences:**</span>

* <span style="font-size: 14px;">**Normalization axis:** BatchNorm normalizes over the batch axis (per feature). LayerNorm normalizes over the feature axis (per example).</span>
* <span style="font-size: 14px;">**Batch dependence:** BatchNorm output depends on other examples in the batch. LayerNorm is batch-independent.</span>
* <span style="font-size: 14px;">**Running statistics:** BatchNorm requires running mean/variance for inference. LayerNorm computes statistics on the fly, behaving identically during training and inference.</span>
* <span style="font-size: 14px;">**Typical domain:** BatchNorm is standard in CNNs (image models). LayerNorm is standard in Transformers (language models).</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">**Ba, Kiros, and Hinton (2016) -- "Layer Normalization."** This paper introduced LayerNorm as an alternative to BatchNorm for recurrent networks. The authors observed that BatchNorm is difficult to apply to RNNs because batch statistics must be computed separately for each time step. LayerNorm solves this by computing statistics over the hidden units within each time step, making it trivially applicable to recurrent architectures.</span>

<span style="font-size: 14px;">**Vaswani et al. (2017) -- "Attention Is All You Need."** The original Transformer adopted LayerNorm in the post-norm configuration. Each sub-layer (self-attention, cross-attention, feed-forward) is followed by a residual connection and LayerNorm.</span>

<span style="font-size: 14px;">**Radford et al. (2019) -- "Language Models are Unsupervised Multitask Learners" (GPT-2).** GPT-2 moved LayerNorm to the pre-norm position and added a final LayerNorm after the last Transformer block. The paper notes that "layer normalization was moved to the input of each sub-block." This change, combined with a modified weight initialization that scales residual layer weights by $1/\sqrt{N}$ where $N$ is the number of residual layers, enabled stable training of GPT-2 Large (36 layers) and GPT-2 XL (48 layers).</span>

<span style="font-size: 14px;">**Xiong et al. (2020) -- "On Layer Normalization in the Transformer Architecture."** This paper provided theoretical analysis showing that pre-norm Transformers have well-behaved gradients at initialization, while post-norm Transformers require warmup because their gradients can be large at initialization. The analysis confirmed what GPT-2 and subsequent models had already adopted in practice.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider a simplified hidden vector with $d = 4$ dimensions. Let $x = [2.0, -1.0, 0.0, 3.0]$, with $\gamma = [1.0, 1.0, 1.0, 1.0]$, $\beta = [0.0, 0.0, 0.0, 0.0]$, and $\epsilon = 10^{-5}$.</span>

<span style="font-size: 14px;">**Step 1 -- Compute the mean:**</span>

$$
\mu = \frac{2.0 + (-1.0) + 0.0 + 3.0}{4} = \frac{4.0}{4} = 1.0
$$

<span style="font-size: 14px;">**Step 2 -- Compute the variance:**</span>

$$
\sigma^2 = \frac{(2.0 - 1.0)^2 + (-1.0 - 1.0)^2 + (0.0 - 1.0)^2 + (3.0 - 1.0)^2}{4}
$$

$$
= \frac{1.0 + 4.0 + 1.0 + 4.0}{4} = \frac{10.0}{4} = 2.5
$$

<span style="font-size: 14px;">**Step 3 -- Compute the standard deviation:**</span>

$$
\sqrt{\sigma^2 + \epsilon} = \sqrt{2.5 + 0.00001} = \sqrt{2.50001} \approx 1.58114
$$

<span style="font-size: 14px;">**Step 4 -- Normalize each element:**</span>

$$
\hat{x}_1 = \frac{2.0 - 1.0}{1.58114} = \frac{1.0}{1.58114} \approx 0.6325
$$

$$
\hat{x}_2 = \frac{-1.0 - 1.0}{1.58114} = \frac{-2.0}{1.58114} \approx -1.2649
$$

$$
\hat{x}_3 = \frac{0.0 - 1.0}{1.58114} = \frac{-1.0}{1.58114} \approx -0.6325
$$

$$
\hat{x}_4 = \frac{3.0 - 1.0}{1.58114} = \frac{2.0}{1.58114} \approx 1.2649
$$

<span style="font-size: 14px;">**Step 5 -- Scale and shift.** With $\gamma = [1, 1, 1, 1]$ and $\beta = [0, 0, 0, 0]$, the output equals the normalized values:</span>

$$
\text{LayerNorm}(x) = [0.6325, -1.2649, -0.6325, 1.2649]
$$

<span style="font-size: 14px;">**Verification.** The output sums to zero (mean = 0) and has unit variance, confirming normalization is correct.</span>

<span style="font-size: 14px;">Now suppose the network has learned $\gamma = [0.5, 2.0, 0.5, 2.0]$ and $\beta = [0.1, -0.1, 0.1, -0.1]$. Applying the affine transformation to each element $y_i = \gamma_i \cdot \hat{x}_i + \beta_i$:</span>

$$
y = [0.5 \times 0.6325 + 0.1, \quad 2.0 \times (-1.2649) - 0.1, \quad 0.5 \times (-0.6325) + 0.1, \quad 2.0 \times 1.2649 - 0.1]
$$

$$
= [0.4163, \quad -2.6298, \quad -0.2163, \quad 2.4298]
$$

<span style="font-size: 14px;">The learned $\gamma$ and $\beta$ allow the network to re-scale and re-center each dimension independently. Dimensions with large $\gamma$ values become more influential in downstream computations, while dimensions with small $\gamma$ values are suppressed.</span>

---

## <span style="font-size: 16px;">Modern Context</span>

<span style="font-size: 14px;">While standard LayerNorm remains widely used, recent architectures have introduced variants that improve efficiency.</span>

<span style="font-size: 14px;">**RMSNorm (Zhang and Sennrich, 2019).** Root Mean Square Layer Normalization drops the mean-centering step entirely, normalizing only by the root mean square of the activations:</span>

$$
\text{RMSNorm}(x)_i = \gamma_i \cdot \frac{x_i}{\sqrt{\frac{1}{d}\sum_{j=1}^{d} x_j^2 + \epsilon}}
$$

<span style="font-size: 14px;">RMSNorm has no $\beta$ parameter and does not subtract the mean. The authors showed this is sufficient for stable training while being computationally cheaper, saving roughly 10-15% of the normalization cost. LLaMA, LLaMA 2, Gemma, and Mistral all use RMSNorm instead of standard LayerNorm. Its success suggests that the primary benefit of normalization comes from controlling the scale of activations, not from centering them.</span>

<span style="font-size: 14px;">**QK-Norm.** Some architectures (notably Gemma and ViT-22B) apply RMSNorm specifically to query and key vectors before computing attention scores, preventing attention logits from growing excessively large.</span>

<span style="font-size: 14px;">**DeepNorm (Wang et al., 2022).** For very deep Transformers (up to 1,000 layers), DeepNorm modifies the residual to $x + \alpha \cdot \text{SubLayer}(\text{LayerNorm}(x))$ with depth-dependent $\alpha$, keeping gradients bounded.</span>

---

## <span style="font-size: 16px;">Pitfalls and Common Mistakes</span>

* <span style="font-size: 14px;">**Normalizing over the wrong dimension.** LayerNorm normalizes over the last dimension (the feature/hidden dimension). Given a tensor of shape $(B, T, d)$, the normalization happens over $d$. Normalizing over $B$ gives BatchNorm; normalizing over $T$ gives InstanceNorm. Both are incorrect for Transformers.</span>

* <span style="font-size: 14px;">**Setting epsilon too small or too large.** The standard value is $\epsilon = 10^{-5}$. Setting it too small (e.g., $10^{-12}$) risks NaN values when variance is near zero. Setting it too large (e.g., $10^{-1}$) biases the normalization.</span>

* <span style="font-size: 14px;">**Confusing LayerNorm with BatchNorm.** BatchNorm requires running mean and variance statistics tracked during training and used at inference. LayerNorm has no running statistics and behaves identically in training and eval modes.</span>

* <span style="font-size: 14px;">**Pre-norm vs post-norm placement errors.** In GPT-2's pre-norm, the residual bypasses the normalized sub-layer: $h = x + \text{SubLayer}(\text{LayerNorm}(x))$. A common mistake is writing $h = \text{LayerNorm}(x + \text{SubLayer}(x))$, which is post-norm and changes gradient flow properties.</span>

* <span style="font-size: 14px;">**Forgetting the final LayerNorm.** GPT-2 applies a final LayerNorm after the last Transformer block. Omitting this changes the scale of inputs to the output projection, affecting logit magnitudes.</span>

* <span style="font-size: 14px;">**Using Bessel's correction.** LayerNorm uses population variance (dividing by $d$), not sample variance (dividing by $d - 1$). Using sample variance can cause mismatches when loading pretrained weights.</span>

* <span style="font-size: 14px;">**Ignoring parameter initialization.** The $\gamma$ and $\beta$ parameters must be initialized to ones and zeros respectively. Random initialization breaks the "identity at init" property and can destabilize early training.</span>

---
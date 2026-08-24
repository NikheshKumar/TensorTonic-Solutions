# <span style="font-size: 20px;">GELU Activation</span>

<span style="font-size: 14px;">The Gaussian Error Linear Unit (GELU) is the activation function used in the feed-forward sublayers of GPT-2 (Radford et al., 2019). Unlike ReLU, which applies a hard zero/identity gate based on sign, GELU applies a smooth, probabilistic gate that weights each element by the probability that it exceeds other inputs drawn from a standard normal distribution. This soft gating preserves gradient flow through negative values and has become the default activation in most modern large language models.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">GELU is an element-wise activation function. Given a scalar input $x$, GELU computes a weighted version of $x$ where the weight is determined by how likely $x$ is to be "large" relative to a standard Gaussian distribution. For large positive $x$, the output is approximately $x$ (identity). For large negative $x$, the output is approximately $0$. For values near zero, the output transitions smoothly between these extremes.</span>

<span style="font-size: 14px;">The function was introduced by Hendrycks and Gimpel in 2016 as a principled alternative to ReLU. Rather than making a binary decision at $x = 0$, GELU blends the input with zero according to a probability derived from the cumulative distribution function of the standard normal. In GPT-2, GELU replaces ReLU in the feed-forward sublayer: $\text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 x + b_1) + b_2$, where $W_1$ projects from $d_{\text{model}}$ to $4 \cdot d_{\text{model}}$ and $W_2$ projects back.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

### <span style="font-size: 14px;">Exact GELU Formula</span>

<span style="font-size: 14px;">The exact definition of GELU is:</span>

$$
\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left(1 + \text{erf}\!\left(\frac{x}{\sqrt{2}}\right)\right)
$$

<span style="font-size: 14px;">Here $\Phi(x)$ is the cumulative distribution function (CDF) of the standard normal distribution, and $\text{erf}$ is the Gauss error function. The factor $\Phi(x)$ acts as a soft gate that ranges continuously from $0$ to $1$.</span>

### <span style="font-size: 14px;">The Error Function</span>

<span style="font-size: 14px;">The error function is a special function defined by an integral:</span>

$$
\text{erf}(z) = \frac{2}{\sqrt{\pi}} \int_0^z e^{-t^2} \, dt
$$

<span style="font-size: 14px;">Key properties: $\text{erf}$ is an odd function ($\text{erf}(-z) = -\text{erf}(z)$), satisfies $\text{erf}(0) = 0$, and approaches $\pm 1$ as $z \to \pm\infty$. The connection to the normal CDF is $\Phi(x) = \frac{1}{2}(1 + \text{erf}(x/\sqrt{2}))$. NumPy, SciPy, and PyTorch all provide optimized $\text{erf}$ implementations accurate to machine precision.</span>

### <span style="font-size: 14px;">Approximate GELU Formula</span>

<span style="font-size: 14px;">Because $\text{erf}$ involves an integral with no closed-form elementary solution, Hendrycks and Gimpel proposed a fast approximation using the hyperbolic tangent:</span>

$$
\text{GELU}_{\text{approx}}(x) = 0.5 \, x \left(1 + \tanh\!\left(\sqrt{\frac{2}{\pi}}\left(x + 0.044715 \, x^3\right)\right)\right)
$$

<span style="font-size: 14px;">The constant $0.044715$ was fitted to minimize the maximum absolute error against the exact GELU. The approximation is accurate to roughly 4-5 decimal places across the entire real line. This tanh-based form is computationally cheaper on hardware that lacks a native $\text{erf}$ instruction. GPT-2's original OpenAI implementation uses this approximate variant.</span>

### <span style="font-size: 14px;">Comparison with ReLU and Sigmoid Gating</span>

<span style="font-size: 14px;">ReLU applies a hard gate: $\text{ReLU}(x) = x \cdot \mathbf{1}_{x > 0}$, where the gate is exactly $0$ or $1$ with a discontinuous switch at $x = 0$. GELU replaces this indicator function with $\Phi(x)$, a smooth S-shaped curve. The sigmoid-weighted linear unit (SiLU), also called Swish, uses the logistic sigmoid instead: $\text{SiLU}(x) = x \cdot \sigma(x)$. GELU and SiLU are numerically close but differ in the exact shape of their gating curve.</span>

---

## <span style="font-size: 16px;">Why GELU Over ReLU</span>

### <span style="font-size: 14px;">Smooth Gating</span>

<span style="font-size: 14px;">ReLU has a non-differentiable kink at $x = 0$ where its derivative jumps from $0$ to $1$ discontinuously. GELU is infinitely differentiable everywhere. The smooth transition means small perturbations near zero produce proportional changes in the output and gradient. This stability is valuable in deep networks with hundreds of layers, where sharp nonlinearities amplify gradient noise and make optimization fragile.</span>

### <span style="font-size: 14px;">Probabilistic Interpretation</span>

<span style="font-size: 14px;">GELU has a clean probabilistic motivation. Consider multiplying an input $x$ by a Bernoulli random variable $m \sim \text{Bernoulli}(\Phi(x))$. The expected value of $mx$ is $x \cdot \Phi(x) = \text{GELU}(x)$. Inputs with larger magnitude are more likely to be kept, while inputs near zero have roughly a 50/50 chance of being zeroed out. This is similar to dropout, but the masking probability depends on the value itself rather than being uniform - an adaptive, input-dependent regularization.</span>

### <span style="font-size: 14px;">No Dead Neurons</span>

<span style="font-size: 14px;">A ReLU neuron whose pre-activation consistently falls below zero will always output zero and receive zero gradient - the "dying ReLU" problem. GELU avoids this because $\text{GELU}(x)$ is never exactly zero for finite $x$, and its derivative is never exactly zero. Even for $x = -3$, the GELU output is approximately $-0.004$ and the gradient is approximately $0.017$, allowing the neuron to eventually recover.</span>

---

## <span style="font-size: 16px;">Mechanics</span>

### <span style="font-size: 14px;">How the Error Function Shapes the Gate</span>

<span style="font-size: 14px;">The gating factor $\Phi(x) = \frac{1}{2}(1 + \text{erf}(x/\sqrt{2}))$ is an S-shaped curve centered at $x = 0$ with $\Phi(0) = 0.5$. For $x = -3$, $\Phi(x) \approx 0.0013$, so the gate is nearly shut. For $x = 3$, $\Phi(x) \approx 0.9987$, so the gate is nearly fully open. The transition from "mostly off" to "mostly on" occurs roughly over $x \in [-2, 2]$, which corresponds to two standard deviations.</span>

### <span style="font-size: 14px;">Output Range</span>

<span style="font-size: 14px;">Unlike ReLU, which outputs values in $[0, +\infty)$, GELU can produce small negative values. The function has a global minimum of approximately $-0.1700$ at $x \approx -0.7032$. For more negative $x$, the output increases back toward zero as $\Phi(x)$ decays faster than $|x|$ grows. The output range is approximately $[-0.17, +\infty)$. This negative dip means GELU is not monotonically increasing on the negative half, distinguishing it from ReLU, Leaky ReLU, and ELU.</span>

### <span style="font-size: 14px;">Derivative of GELU</span>

<span style="font-size: 14px;">The derivative of GELU with respect to $x$ is obtained via the product rule:</span>

$$
\text{GELU}'(x) = \Phi(x) + x \cdot \phi(x)
$$

<span style="font-size: 14px;">where $\phi(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2}$ is the standard normal PDF. At $x = 0$, the derivative is $0.5$. For large positive $x$, $\Phi(x) \to 1$ and $x \cdot \phi(x) \to 0$, so the derivative approaches $1$. For large negative $x$, both terms approach $0$. The derivative can become slightly negative when $x < 0$ (since $x \cdot \phi(x) < 0$), corresponding to the region where GELU dips below zero.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

### <span style="font-size: 14px;">GPT-2 (Radford et al., 2019)</span>

<span style="font-size: 14px;">GPT-2 ("Language Models are Unsupervised Multitask Learners") is a decoder-only Transformer language model trained on WebText, scaling up to 1.5 billion parameters. One of its modifications relative to the original Transformer is replacing ReLU with GELU in the position-wise feed-forward layers. The paper does not extensively justify this choice, but by 2019 GELU had demonstrated consistent improvements over ReLU in language modeling benchmarks.</span>

### <span style="font-size: 14px;">Hendrycks and Gimpel (2016)</span>

<span style="font-size: 14px;">The GELU activation was formally introduced in "Gaussian Error Linear Units (GELUs)" by Dan Hendrycks and Kevin Gimpel. The paper argues that an activation function should blend both the nonlinear transformation role and the stochastic regularization role (randomly masking activations, as dropout does). The authors showed GELU outperformed ReLU, ELU, and SELU on MNIST, CIFAR-10, CIFAR-100, SVHN, and a language modeling benchmark. Improvements were consistent but modest (typically 0.1-0.5% accuracy), making GELU a reliable incremental improvement.</span>

### <span style="font-size: 14px;">Adoption in BERT and Beyond</span>

<span style="font-size: 14px;">BERT (Devlin et al., 2019) also uses GELU in its feed-forward sublayers, making it one of the first high-profile models to adopt the activation. After GPT-2 and BERT, GELU became the near-universal default in Transformer-based language models. GPT-3, RoBERTa, ALBERT, DeBERTa, and many others all use GELU. Some later models switched to SiLU/Swish (LLaMA, PaLM) or GLU variants (SwiGLU in LLaMA 2, GEGLU), but GELU remains the most common single activation function in production language models.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Computing $\text{GELU}(x)$ for $x \in [-2, -1, 0, 0.5, 1, 2]$ using the exact formula. For each value, the steps are: compute $x / \sqrt{2}$, evaluate $\text{erf}$, compute the gate $\Phi(x) = 0.5(1 + \text{erf}(x/\sqrt{2}))$, then multiply $x \cdot \Phi(x)$.</span>

### <span style="font-size: 14px;">$x = -2$</span>

<span style="font-size: 14px;">$x / \sqrt{2} = -1.4142$, so $\text{erf}(-1.4142) = -0.9545$. Gate: $\Phi(-2) = 0.5 \cdot (1 - 0.9545) = 0.0228$. Result: $\text{GELU}(-2) = (-2)(0.0228) = -0.0455$. The gate is nearly closed, so the input is almost completely suppressed but not exactly zero.</span>

### <span style="font-size: 14px;">$x = -1$</span>

<span style="font-size: 14px;">$x / \sqrt{2} = -0.7071$, so $\text{erf}(-0.7071) = -0.6827$. Gate: $\Phi(-1) = 0.5 \cdot (1 - 0.6827) = 0.1587$. Result: $\text{GELU}(-1) = (-1)(0.1587) = -0.1587$. About $15.9\%$ of the signal passes through. The negative output illustrates the characteristic dip of GELU.</span>

### <span style="font-size: 14px;">$x = 0$</span>

<span style="font-size: 14px;">$x / \sqrt{2} = 0$, so $\text{erf}(0) = 0$. Gate: $\Phi(0) = 0.5$. Result: $\text{GELU}(0) = 0 \cdot 0.5 = 0$. The gate is exactly $0.5$, but the input is zero, so the output is zero regardless.</span>

### <span style="font-size: 14px;">$x = 0.5$</span>

<span style="font-size: 14px;">$x / \sqrt{2} = 0.3536$, so $\text{erf}(0.3536) = 0.3829$. Gate: $\Phi(0.5) = 0.5 \cdot 1.3829 = 0.6915$. Result: $\text{GELU}(0.5) = 0.5 \cdot 0.6915 = 0.3457$. Compare with $\text{ReLU}(0.5) = 0.5$: GELU outputs only $69.1\%$ of the input because the gate is not yet fully open.</span>

### <span style="font-size: 14px;">$x = 1$</span>

<span style="font-size: 14px;">$x / \sqrt{2} = 0.7071$, so $\text{erf}(0.7071) = 0.6827$. Gate: $\Phi(1) = 0.5 \cdot 1.6827 = 0.8413$. Result: $\text{GELU}(1) = 1 \cdot 0.8413 = 0.8413$. ReLU would output $1.0$, so GELU attenuates positive inputs not yet far from zero.</span>

### <span style="font-size: 14px;">$x = 2$</span>

<span style="font-size: 14px;">$x / \sqrt{2} = 1.4142$, so $\text{erf}(1.4142) = 0.9545$. Gate: $\Phi(2) = 0.5 \cdot 1.9545 = 0.9772$. Result: $\text{GELU}(2) = 2 \cdot 0.9772 = 1.9545$. The gate is $97.7\%$ open, so the output is very close to the identity. By $x = 3$, the gate exceeds $99.8\%$ and GELU becomes virtually indistinguishable from the identity function.</span>

---

## <span style="font-size: 16px;">Modern Context</span>

### <span style="font-size: 14px;">GELU Variants</span>

<span style="font-size: 14px;">In PyTorch, `torch.nn.GELU(approximate='none')` uses the exact $\text{erf}$-based formula, while `torch.nn.GELU(approximate='tanh')` uses the tanh approximation. The original GPT-2 codebase used the tanh variant. Modern frameworks default to the exact form because contemporary GPU hardware has fast $\text{erf}$ implementations. When reproducing GPT-2 weights, using the wrong variant causes numerical mismatches that accumulate across layers.</span>

### <span style="font-size: 14px;">SiLU/Swish</span>

<span style="font-size: 14px;">The Sigmoid Linear Unit (SiLU), also known as Swish (Ramachandran et al., 2017), replaces the normal CDF gate with the logistic sigmoid: $\text{SiLU}(x) = x \cdot \sigma(x)$ where $\sigma(x) = 1/(1 + e^{-x})$. LLaMA (Touvron et al., 2023) and PaLM (Chowdhery et al., 2022) use SiLU-based activations. The choice between GELU and SiLU is often determined by ecosystem conventions rather than empirical superiority.</span>

### <span style="font-size: 14px;">GLU-Based Activations</span>

<span style="font-size: 14px;">Gated Linear Units (GLU) split the intermediate representation and use one half to gate the other: $\text{GLU}(x) = (W_1 x) \odot \sigma(W_2 x)$. Shazeer (2020) proposed GEGLU ($\text{GELU}$ gate), SwiGLU ($\text{SiLU}$ gate), and ReGLU ($\text{ReLU}$ gate). SwiGLU has become popular in LLaMA 2, Mistral, and Gemma. These variants use two projection matrices, so models typically reduce $d_{\text{ff}}$ from $4 \cdot d_{\text{model}}$ to $\frac{8}{3} \cdot d_{\text{model}}$ to keep total parameters comparable.</span>

### <span style="font-size: 14px;">GELU in Modern LLMs</span>

<span style="font-size: 14px;">GELU remains the activation of choice in GPT-4, BERT-family models, and many encoder-decoder architectures. The decoder-only LLM ecosystem has split between GELU (GPT-series) and SwiGLU (LLaMA-family). Both perform comparably, and the choice is dictated by which pretrained model family a practitioner builds on rather than by activation benchmarks.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Using the approximate formula when the exact formula is required.** The tanh approximation differs from exact GELU by up to $5 \times 10^{-4}$. When loading pretrained GPT-2 weights, using the wrong variant produces accumulated errors across 48 layers (in GPT-2 XL) that can shift output logits enough to change top-k predictions. Always verify which variant the pretrained model was trained with.</span>

* <span style="font-size: 14px;">**Confusing GELU with ReLU behavior at negative inputs.** A common misconception is that GELU outputs zero for all negative inputs. In reality, GELU outputs small negative values for moderately negative inputs, with a minimum of approximately $-0.17$ near $x = -0.7$. Code that clips GELU outputs to $[0, \infty)$ or asserts non-negativity will silently corrupt the computation.</span>

* <span style="font-size: 14px;">**Numerical issues with erf for extreme inputs.** Naive $\text{erf}$ implementations using a truncated Taylor series lose precision for $|x| > 6$. Production code should use `math.erf`, `np.erf`, or `torch.erf`, which use rational Chebyshev approximations accurate across the entire input range.</span>

* <span style="font-size: 14px;">**Forgetting that GELU is not monotonic on the negative side.** GELU decreases from $0$ to $-0.17$ as $x$ goes from $0$ to $-0.7$, then increases back toward $0$. The derivative is negative near $x \approx -0.7$. Optimizations that assume monotonicity (such as pruning heuristics assuming "larger pre-activation implies larger output") will be incorrect for GELU.</span>

* <span style="font-size: 14px;">**Mixing up GELU and SiLU/Swish in model implementations.** GELU uses the normal CDF $\Phi(x)$ as the gate; SiLU uses the logistic sigmoid $\sigma(x)$. The two functions differ most noticeably in the range $|x| \in [1, 3]$. When porting a model between frameworks, accidentally swapping these activations will not crash the model but will degrade performance, especially for fine-tuned models where the weights have adapted to the specific activation's shape.</span>

* <span style="font-size: 14px;">**Ignoring the interaction between GELU and layer normalization.** In GPT-2, LayerNorm precedes the feed-forward sublayer (pre-norm architecture), centering inputs to zero mean. This means GELU frequently receives values near zero, where its nonlinearity is strongest. If LayerNorm is applied incorrectly (wrong axis, missing affine parameters), the input distribution shifts, reducing the effective nonlinearity and degrading model quality silently.</span>

---
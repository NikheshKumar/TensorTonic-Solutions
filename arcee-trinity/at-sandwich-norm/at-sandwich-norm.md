# <span style="font-size: 20px;">Sandwich Norm with Depth Scaling</span>

<span style="font-size: 14px;">Sandwich Norm is a normalization strategy that applies RMSNorm both before and after each sub-layer (attention or FFN), forming a "sandwich" around the computation. The post-norm output is then scaled by a depth-dependent factor before being added back to the residual stream. Arcee Trinity uses this pattern in every transformer block to maintain training stability as the model grows deeper.</span>

<span style="font-size: 14px;">The depth scaling factor ensures that contributions from deeper layers are progressively attenuated, preventing gradient explosion and keeping the residual stream well-conditioned throughout the network.</span>

---

## <span style="font-size: 16px;">What It Is / What It Does</span>

<span style="font-size: 14px;">There are three main normalization placement strategies in transformers:</span>

* <span style="font-size: 14px;">**Pre-Norm (Pre-LN):** Normalization before the sub-layer. The residual adds raw sub-layer output to the input. Used in GPT-2, LLaMA, and most modern LLMs. Stabilizes training but allows sub-layer outputs to grow unchecked.</span>
* <span style="font-size: 14px;">**Post-Norm (Post-LN):** Normalization after the residual addition. The original transformer design (Vaswani et al., 2017). Produces well-conditioned representations but suffers from gradient vanishing in deep networks.</span>
* <span style="font-size: 14px;">**Sandwich Norm:** Normalization both before AND after the sub-layer, but before residual addition. Pre-norm stabilizes the sub-layer input. Post-norm constrains the sub-layer output magnitude.</span>

<span style="font-size: 14px;">Sandwich norm addresses a specific weakness of pre-norm: while pre-norm stabilizes what goes into the sub-layer, it places no constraint on what comes out. In deep networks, sub-layer outputs can grow in magnitude, causing the residual stream to accumulate large values. Adding post-norm bounds the output magnitude before residual addition.</span>

<span style="font-size: 14px;">Depth scaling adds a further refinement. Instead of adding the full post-normalized output, it multiplies by a factor that decreases with depth, preventing the accumulation problem from compounding over hundreds of layers.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">**RMSNorm** is the normalization primitive used at both the pre and post positions. For an input vector $x \in \mathbb{R}^d$ with learnable gain $\gamma \in \mathbb{R}^d$:</span>

$$
\text{RMSNorm}(x, \gamma) = \frac{x \cdot \gamma}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}}
$$

* <span style="font-size: 14px;">**$d$:** Dimension of the input vector.</span>
* <span style="font-size: 14px;">**$\gamma$:** Learnable per-element scale parameter (initialized to ones).</span>
* <span style="font-size: 14px;">**$\epsilon$:** Small constant for numerical stability (typically $10^{-6}$ or $10^{-8}$).</span>

<span style="font-size: 14px;">RMSNorm differs from LayerNorm by omitting mean subtraction. It only divides by the root mean square of the activations, making it computationally cheaper and often equally effective.</span>

<span style="font-size: 14px;">**Sandwich Norm Pattern** for a sub-layer $f$ (either attention or FFN) at layer index $l$:</span>

<span style="font-size: 14px;">Step 1: Pre-normalize the input.</span>

$$
\text{pre\_normed} = \text{RMSNorm}(x, \gamma_{\text{pre}})
$$

<span style="font-size: 14px;">Step 2: Apply the sub-layer.</span>

$$
\text{sub\_out} = f(\text{pre\_normed})
$$

<span style="font-size: 14px;">Step 3: Post-normalize the sub-layer output.</span>

$$
\text{post\_normed} = \text{RMSNorm}(\text{sub\_out}, \gamma_{\text{post}})
$$

<span style="font-size: 14px;">Step 4: Compute the depth-dependent scaling factor.</span>

$$
\alpha_l = \frac{1}{\sqrt{2l + 1}}
$$

* <span style="font-size: 14px;">**$l$:** The layer index (0-indexed). Layer 0 gets $\alpha_0 = 1/\sqrt{1} = 1.0$. Layer 1 gets $\alpha_1 = 1/\sqrt{3} \approx 0.577$. Layer 10 gets $\alpha_{10} = 1/\sqrt{21} \approx 0.218$.</span>

<span style="font-size: 14px;">Step 5: Scale and add to residual.</span>

$$
\text{output} = x + \text{post\_normed} \cdot \alpha_l
$$

<span style="font-size: 14px;">The full equation combining all steps:</span>

$$
\text{output} = x + \frac{\text{RMSNorm}\bigl(f(\text{RMSNorm}(x, \gamma_{\text{pre}})),\; \gamma_{\text{post}}\bigr)}{\sqrt{2l + 1}}
$$

<span style="font-size: 14px;">Note that $\gamma_{\text{pre}}$ and $\gamma_{\text{post}}$ are separate learnable parameters. Each sub-layer has its own pair, so a single transformer block with both attention and FFN has four sets of RMSNorm parameters.</span>

---

## <span style="font-size: 16px;">Why Sandwich Norm</span>

<span style="font-size: 14px;">Standard pre-norm has a subtle but important limitation. Consider a pre-norm residual block:</span>

$$
x_{l+1} = x_l + f(\text{RMSNorm}(x_l))
$$

<span style="font-size: 14px;">The normalization constrains the input to $f$, but places no bound on $f$'s output. Over $L$ layers, the residual stream magnitude can grow as $O(\sqrt{L})$ or worse.</span>

* <span style="font-size: 14px;">**Activation growth:** As the residual stream grows, each new layer's relative contribution shrinks (pre-norm rescales before each sub-layer). Early layers dominate the representation and late layers contribute diminishing signals -- effective depth reduction.</span>
* <span style="font-size: 14px;">**Gradient imbalance:** Large residual magnitudes create uneven gradient scales. Early layers see larger gradients while late layers see smaller ones.</span>
* <span style="font-size: 14px;">**Training instability:** At scale (billions of parameters, hundreds of layers), these effects compound and cause loss spikes or divergence.</span>

<span style="font-size: 14px;">Adding post-norm directly addresses activation growth. The second RMSNorm re-normalizes the sub-layer output before residual addition, ensuring controlled magnitude regardless of what the sub-layer produced internally.</span>

<span style="font-size: 14px;">In Arcee Trinity, sandwich norm provides a structural guarantee that no single sub-layer can inject disproportionately large values into the residual stream, making the architecture robust across different model sizes without per-size tuning of initialization or learning rates.</span>

---

## <span style="font-size: 16px;">Depth Scaling</span>

<span style="font-size: 14px;">The depth scaling factor $\alpha_l = 1/\sqrt{2l + 1}$ is applied to the post-normalized sub-layer output before residual addition. This ensures that deeper layers contribute progressively less to the residual stream.</span>

<span style="font-size: 14px;">**Why $1/\sqrt{2l + 1}$?**</span>

<span style="font-size: 14px;">Consider the residual stream after $L$ layers. Without scaling, the residual is:</span>

$$
x_L = x_0 + \sum_{l=0}^{L-1} g_l
$$

<span style="font-size: 14px;">where $g_l$ is the post-normalized sub-layer output at layer $l$. If each $g_l$ has roughly unit variance (thanks to post-norm), the variance of $x_L$ grows linearly with $L$. With depth scaling:</span>

$$
x_L = x_0 + \sum_{l=0}^{L-1} \frac{g_l}{\sqrt{2l + 1}}
$$

<span style="font-size: 14px;">The sum of squared scales is $\sum_{l=0}^{L-1} \frac{1}{2l+1}$, which grows as $O(\log L)$ rather than $O(L)$. This logarithmic growth keeps the residual stream magnitude from exploding even in very deep networks.</span>

* <span style="font-size: 14px;">**Layer 0:** $\alpha_0 = 1/\sqrt{1} = 1.0$ -- the first layer contributes fully.</span>
* <span style="font-size: 14px;">**Layer 5:** $\alpha_5 = 1/\sqrt{11} \approx 0.302$ -- roughly 30% contribution.</span>
* <span style="font-size: 14px;">**Layer 15:** $\alpha_{15} = 1/\sqrt{31} \approx 0.180$ -- about 18% contribution.</span>
* <span style="font-size: 14px;">**Layer 40:** $\alpha_{40} = 1/\sqrt{81} = 1/9 \approx 0.111$ -- about 11% contribution.</span>

<span style="font-size: 14px;">**Connection to DeepNet.** The DeepNet paper (Wang et al., 2022) introduced depth-dependent scaling for training very deep transformers (up to 1000 layers). DeepNet uses $\alpha = (2N)^{1/4}$ as a constant multiplier on residual connections with modified initialization. The Arcee Trinity approach refines this with a per-layer scaling factor for finer attenuation control.</span>

<span style="font-size: 14px;">**Gradient perspective.** During backpropagation, the gradient through layer $l$ is scaled by $\alpha_l$. Deeper layers receive attenuated gradients, but because they also contribute less in the forward pass, the signal-to-noise ratio remains balanced across layers.</span>

---

## <span style="font-size: 16px;">Paper Context: Arcee Trinity</span>

<span style="font-size: 14px;">Arcee Trinity applies sandwich norm uniformly to every sub-layer in every transformer block. Each block contains two sub-layers: multi-head attention and a feed-forward network (FFN). Both receive the full sandwich treatment:</span>

* <span style="font-size: 14px;">**Attention sub-layer:** $x' = x + \alpha_l \cdot \text{RMSNorm}(\text{Attn}(\text{RMSNorm}(x, \gamma_{\text{pre}}^{\text{attn}})),\; \gamma_{\text{post}}^{\text{attn}})$</span>
* <span style="font-size: 14px;">**FFN sub-layer:** $\text{out} = x' + \alpha_l \cdot \text{RMSNorm}(\text{FFN}(\text{RMSNorm}(x', \gamma_{\text{pre}}^{\text{ffn}})),\; \gamma_{\text{post}}^{\text{ffn}})$</span>

<span style="font-size: 14px;">This means each block has four RMSNorm operations (pre-attn, post-attn, pre-ffn, post-ffn) with four separate sets of learnable $\gamma$ parameters. The depth scale $\alpha_l$ is the same for both sub-layers within a block since they share the same layer index.</span>

<span style="font-size: 14px;">The design rationale for Arcee Trinity is training stability at scale. By combining post-normalization with depth scaling, the architecture achieves several practical benefits:</span>

* <span style="font-size: 14px;">**No warm-up sensitivity:** Sandwich norm with depth scaling is more forgiving of learning rate warm-up schedules because the residual stream stays well-conditioned from the start.</span>
* <span style="font-size: 14px;">**Consistent gradient norms:** Post-norm and depth scaling keep gradient norms relatively uniform across layers, reducing the need for aggressive gradient clipping.</span>
* <span style="font-size: 14px;">**Scalable depth:** Logarithmic residual variance growth means the architecture can be made deeper without retuning hyperparameters.</span>
* <span style="font-size: 14px;">**Minimal overhead:** RMSNorm is cheaper than LayerNorm (no mean computation), and the depth scale is a single scalar multiply -- negligible compared to attention or FFN computation.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider a 4-dimensional hidden state with $x = [2.0, -1.0, 0.5, 1.5]$, $\gamma_{\text{pre}} = \gamma_{\text{post}} = [1, 1, 1, 1]$, and $\epsilon = 10^{-6}$.</span>

<span style="font-size: 14px;">**Step 1: Pre-RMSNorm.**</span>

$$
\text{mean}(x^2) = \frac{4.0 + 1.0 + 0.25 + 2.25}{4} = 1.875
$$

$$
\text{RMS} = \sqrt{1.875 + 10^{-6}} \approx 1.3693
$$

$$
\text{pre\_normed} = \frac{[2.0, -1.0, 0.5, 1.5]}{1.3693} \approx [1.461, -0.730, 0.365, 1.095]
$$

<span style="font-size: 14px;">**Step 2: Sub-layer.** Suppose $f(\text{pre\_normed}) = [0.8, -2.5, 1.2, 3.0]$.</span>

<span style="font-size: 14px;">**Step 3: Post-RMSNorm.**</span>

$$
\text{mean}(\text{sub\_out}^2) = \frac{0.64 + 6.25 + 1.44 + 9.0}{4} = 4.3325
$$

$$
\text{RMS} = \sqrt{4.3325} \approx 2.0815
$$

$$
\text{post\_normed} = \frac{[0.8, -2.5, 1.2, 3.0]}{2.0815} \approx [0.384, -1.201, 0.577, 1.441]
$$

<span style="font-size: 14px;">The post-norm brought the magnitude under control -- the sub-layer output had elements as large as 3.0, but post-norm rescales them.</span>

<span style="font-size: 14px;">**Step 4: Depth scaling at layer 0 ($l = 0$).**</span>

$$
\alpha_0 = \frac{1}{\sqrt{1}} = 1.0
$$

$$
\text{output}_0 = [2.0, -1.0, 0.5, 1.5] + [0.384, -1.201, 0.577, 1.441] = [2.384, -2.201, 1.077, 2.941]
$$

<span style="font-size: 14px;">**Step 5: Depth scaling at layer 4 ($l = 4$).**</span>

$$
\alpha_4 = \frac{1}{\sqrt{9}} = 0.333
$$

$$
\text{scaled} = [0.384, -1.201, 0.577, 1.441] \cdot 0.333 = [0.128, -0.400, 0.192, 0.480]
$$

$$
\text{output}_4 = [2.0, -1.0, 0.5, 1.5] + [0.128, -0.400, 0.192, 0.480] = [2.128, -1.400, 0.692, 1.980]
$$

<span style="font-size: 14px;">At layer 0, the sub-layer shifted the residual by as much as 1.44. At layer 4, the same output only shifted the residual by 0.48. Deeper layers produce smaller perturbations to the residual stream.</span>

---

## <span style="font-size: 16px;">Variants and Modern Context</span>

<span style="font-size: 14px;">Sandwich norm sits within a broader landscape of normalization strategies in transformers:</span>

* <span style="font-size: 14px;">**Post-LN (Original Transformer, Vaswani et al., 2017):** Norm after residual addition. Well-conditioned layer outputs, but gradients vanish in deep networks. BERT also used Post-LN.</span>
* <span style="font-size: 14px;">**Pre-LN (GPT-2, Radford et al., 2019):** Norm before the sub-layer. Default for most large LLMs (GPT-3, LLaMA, Mistral). Stable training but allows residual growth.</span>
* <span style="font-size: 14px;">**DeepNet (Wang et al., 2022):** Post-LN with constant depth scaling $\alpha = (2N)^{1/4}$ and modified Xavier initialization. Trains transformers up to 1000 layers. The depth-scaling insight directly informs Arcee Trinity.</span>
* <span style="font-size: 14px;">**QK-Norm:** Normalizes query and key vectors before the attention dot product, preventing large logits. Orthogonal to sandwich norm and can be combined with it.</span>
* <span style="font-size: 14px;">**CogView Sandwich Norm (Ding et al., 2021):** Introduced the "sandwich" terminology -- LayerNorm before and after each sub-layer for stable vision-language training. Arcee Trinity adapts this with RMSNorm and adds per-layer depth scaling.</span>
* <span style="font-size: 14px;">**Sub-LN (Foundation Transformers, Wang et al., 2022):** Norm both before the sub-layer and before residual addition, similar to sandwich norm. Enables transferability across tasks and modalities.</span>

<span style="font-size: 14px;">The trend in modern architectures is toward combining multiple normalization strategies. Sandwich norm with depth scaling takes the stability of pre-norm, adds output control from post-norm, and uses depth scaling to manage the cumulative effect across layers.</span>

---

## <span style="font-size: 16px;">Common Pitfalls</span>

* <span style="font-size: 14px;">**Forgetting the post-norm:** Implementing only pre-norm and depth scaling but omitting post-norm. Without post-norm, depth scaling operates on unnormalized outputs that can have arbitrary magnitude.</span>
* <span style="font-size: 14px;">**Wrong depth scale formula:** Using $1/\sqrt{2l}$ instead of $1/\sqrt{2l + 1}$. At layer 0, $1/\sqrt{0}$ is undefined. The $+1$ ensures validity for all layers starting from $l = 0$.</span>
* <span style="font-size: 14px;">**Applying depth scale to the residual instead of sub-layer output:** Writing $\text{output} = \alpha_l \cdot (x + \text{post\_normed})$ instead of $\text{output} = x + \alpha_l \cdot \text{post\_normed}$. The first version scales the entire residual stream, destroying information from earlier layers.</span>
* <span style="font-size: 14px;">**Epsilon too small in RMSNorm:** Using $\epsilon = 0$ or $10^{-12}$ causes numerical instability in half-precision (fp16/bf16). The denominator can underflow to NaN. Safe choices: $10^{-6}$ for fp32, $10^{-5}$ for bf16.</span>
* <span style="font-size: 14px;">**Confusing layer_idx with total layers:** Depth scale uses the layer index $l$ (which layer am I?), not the total number of layers $L$. Using $L$ produces a constant scale for all layers, defeating per-layer attenuation.</span>
* <span style="font-size: 14px;">**Sharing gamma parameters between pre and post norms:** Pre-norm and post-norm must have independent $\gamma$ parameters. They normalize different distributions (residual stream vs. sub-layer output), so tying them degrades performance.</span>
* <span style="font-size: 14px;">**Applying depth scale only during training or only during inference:** The depth scale is a fixed architectural component, not regularization. It must be present in both training and inference, unlike dropout.</span>

---
# <span style="font-size: 20px;">Position-wise Feed-Forward Network</span>

<span style="font-size: 14px;">The position-wise feed-forward network (FFN) is a two-layer MLP that sits inside every Transformer block, applied independently to each token position. In GPT-2, it follows $\text{FFN}(x) = \text{GELU}(xW_1^T + b_1)W_2^T + b_2$, expanding the hidden dimension by 4x before projecting it back down. Despite being structurally simple, the FFN accounts for the majority of parameters in each Transformer block and provides the nonlinear computation capacity that attention alone cannot.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">The FFN is a two-layer multilayer perceptron applied at every position in the sequence. For each position's hidden state $x \in \mathbb{R}^d$, it expands to dimension $4d$ through the first linear layer, applies GELU, then projects back to $d$ through the second linear layer:</span>

* <span style="font-size: 14px;">**First linear layer:** project from $d$ to $4d$, producing an intermediate representation in a higher-dimensional space</span>
* <span style="font-size: 14px;">**GELU activation:** apply a smooth, non-monotonic nonlinearity element-wise to the intermediate representation</span>
* <span style="font-size: 14px;">**Second linear layer:** project from $4d$ back to $d$, compressing the result back to the model's hidden dimension</span>

<span style="font-size: 14px;">The FFN uses the same weights at every position -- $W_1$ and $W_2$ are shared across all tokens. This is what "position-wise" means: each position is treated as an independent input, with no information mixing between positions. In GPT-2's Transformer block, the FFN is the second sub-layer, computing $x + \text{FFN}(\text{LN}(x))$ with a residual connection.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">The complete FFN computation in GPT-2 is:</span>

$$
\text{FFN}(x) = \text{GELU}(xW_1^T + b_1)W_2^T + b_2
$$

<span style="font-size: 14px;">where:</span>

* <span style="font-size: 14px;">$x \in \mathbb{R}^d$ is the input hidden state at a single position</span>
* <span style="font-size: 14px;">$W_1 \in \mathbb{R}^{4d \times d}$ is the weight matrix for the first linear layer (expansion)</span>
* <span style="font-size: 14px;">$b_1 \in \mathbb{R}^{4d}$ is the bias vector for the first linear layer</span>
* <span style="font-size: 14px;">$W_2 \in \mathbb{R}^{d \times 4d}$ is the weight matrix for the second linear layer (contraction)</span>
* <span style="font-size: 14px;">$b_2 \in \mathbb{R}^d$ is the bias vector for the second linear layer</span>

<span style="font-size: 14px;">**Step 1 -- Expand.** Compute the intermediate hidden state:</span>

$$
h = xW_1^T + b_1, \quad h \in \mathbb{R}^{4d}
$$

<span style="font-size: 14px;">**Step 2 -- Activate.** Apply the GELU activation element-wise:</span>

$$
a = \text{GELU}(h), \quad a_i = h_i \cdot \Phi(h_i)
$$

<span style="font-size: 14px;">where $\Phi(\cdot)$ is the standard Gaussian CDF. GPT-2 uses the $\tanh$ approximation:</span>

$$
\text{GELU}(x) \approx 0.5 \cdot x \cdot \left(1 + \tanh\!\left[\sqrt{\frac{2}{\pi}}\left(x + 0.044715 x^3\right)\right]\right)
$$

<span style="font-size: 14px;">**Step 3 -- Contract.** Project back to the model dimension:</span>

$$
\text{FFN}(x) = aW_2^T + b_2, \quad \text{FFN}(x) \in \mathbb{R}^d
$$

<span style="font-size: 14px;">The output has the same dimension $d$ as the input, which is necessary for the residual connection in the Transformer block.</span>

---

## <span style="font-size: 16px;">Why 4x Expansion</span>

<span style="font-size: 14px;">The FFN expands the hidden dimension by a factor of 4 before contracting it back. This expand-contract bottleneck is a deliberate design choice from "Attention Is All You Need" (Vaswani et al., 2017), and GPT-2 follows it exactly.</span>

<span style="font-size: 14px;">The reason is **computational capacity**. Self-attention computes weighted averages of value vectors -- linear combinations that cannot approximate arbitrary functions. The FFN, by expanding into a higher-dimensional space and applying a nonlinearity, gives the model capacity for complex, nonlinear transformations of the features produced by attention.</span>

* <span style="font-size: 14px;">**Too small (1x-2x):** The intermediate space cannot form rich representations, creating a capacity bottleneck.</span>
* <span style="font-size: 14px;">**Too large (8x+):** Parameters and FLOPs grow proportionally with diminishing returns on quality.</span>
* <span style="font-size: 14px;">**4x is empirically optimal:** Vaswani et al. found $d_{\text{ff}} = 4 \times d_{\text{model}}$ consistently worked well, adopted by GPT-2, BERT, T5, and many others.</span>

<span style="font-size: 14px;">The nonlinearity is essential: without GELU, the two linear layers collapse into one equivalent transformation $xW_1^TW_2^T$, making the expansion pointless. The activation in the high-dimensional intermediate space lets the FFN selectively activate different neurons for different input patterns.</span>

---

## <span style="font-size: 16px;">Position-Wise: No Cross-Position Interaction</span>

<span style="font-size: 14px;">"Position-wise" means the FFN processes each token position completely independently. For $T$ tokens, the same function with the same weights is applied to each position separately. This creates a critical separation of concerns:</span>

* <span style="font-size: 14px;">**Attention handles cross-position interaction.** Each position gathers information from other positions, producing a weighted mixture of context.</span>
* <span style="font-size: 14px;">**FFN handles per-position computation.** It transforms the attended representation through a nonlinear function -- where the model "thinks" about gathered information.</span>

<span style="font-size: 14px;">In tensor terms, for $X \in \mathbb{R}^{B \times T \times d}$, the FFN applies the same operation along the last dimension at every $(b, t)$ position -- equivalent to reshaping to $(B \cdot T, d)$, applying the MLP, and reshaping back. Research has shown that FFN layers act as key-value memories: $W_1$'s rows are "keys" matching input patterns; $W_2$'s columns are "values" storing associated information.</span>

---

## <span style="font-size: 16px;">Why GELU Instead of ReLU</span>

<span style="font-size: 14px;">The original Transformer used ReLU ($\max(0, x)$) in the FFN. GPT-2 replaced it with GELU (Gaussian Error Linear Unit, Hendrycks & Gimpel, 2016), defined as:</span>

$$
\text{GELU}(x) = x \cdot \Phi(x) = x \cdot P(Z \leq x), \quad Z \sim \mathcal{N}(0, 1)
$$

<span style="font-size: 14px;">This multiplies each input by the probability that a standard normal variable is less than that input. The result behaves like ReLU for large magnitudes but transitions smoothly near zero:</span>

* <span style="font-size: 14px;">**Large positive $x$:** $\Phi(x) \approx 1$, so $\text{GELU}(x) \approx x$ (identity, like ReLU)</span>
* <span style="font-size: 14px;">**Large negative $x$:** $\Phi(x) \approx 0$, so $\text{GELU}(x) \approx 0$ (suppressed, like ReLU)</span>
* <span style="font-size: 14px;">**Near zero:** smooth transition instead of ReLU's hard cutoff; small negative values are softly suppressed rather than zeroed out</span>

<span style="font-size: 14px;">Why this matters for training:</span>

* <span style="font-size: 14px;">**No dead neurons:** ReLU kills any neuron whose pre-activation is always negative -- zero gradient, no recovery. GELU keeps all neurons alive with nonzero gradients.</span>
* <span style="font-size: 14px;">**Smooth gradients:** ReLU has a discontinuous gradient at $x = 0$. GELU is smooth everywhere, making optimization more stable.</span>
* <span style="font-size: 14px;">**Implicit regularization:** GELU acts as a deterministic soft dropout -- randomly applying identity or zero based on input magnitude.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">The FFN was introduced in "Attention Is All You Need" (Vaswani et al., 2017) as one of two sub-layers per Transformer block. The original paper used ReLU with $d_{\text{ff}} = 2048$ for $d_{\text{model}} = 512$, establishing the 4x ratio and describing it as "two linear transformations with a ReLU activation in between."</span>

<span style="font-size: 14px;">GPT-2 (Radford et al., 2019) kept the 4x ratio but replaced ReLU with GELU across all model sizes:</span>

* <span style="font-size: 14px;">**GPT-2 Small:** $d = 768$, $d_{\text{ff}} = 3072$, 12 layers</span>
* <span style="font-size: 14px;">**GPT-2 Medium:** $d = 1024$, $d_{\text{ff}} = 4096$, 24 layers</span>
* <span style="font-size: 14px;">**GPT-2 Large:** $d = 1280$, $d_{\text{ff}} = 5120$, 36 layers</span>
* <span style="font-size: 14px;">**GPT-2 XL:** $d = 1600$, $d_{\text{ff}} = 6400$, 48 layers</span>

<span style="font-size: 14px;">The FFN provides the model's "memory" or "knowledge storage" capacity. While attention routes information between positions, the FFN stores factual associations. Studies show that specific FFN neurons activate for specific knowledge, and editing FFN weights can modify factual outputs without retraining. Despite conceptual simplicity, the FFN accounts for roughly two-thirds of each block's parameters.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Trace through the FFN with $d = 4$ and expansion ratio 4, so $d_{\text{ff}} = 16$. For clarity, we trace the first 4 of the 16 intermediate neurons.</span>

<span style="font-size: 14px;">**Input:** $x = [0.5, -0.3, 0.8, -0.1]$.</span>

<span style="font-size: 14px;">**Step 1 -- Expand via $W_1$.** Using the first 4 rows of $W_1 \in \mathbb{R}^{16 \times 4}$ and $b_1$:</span>

$$
W_1^{[0:4]} = \begin{pmatrix} 0.2 & 0.5 & -0.3 & 0.1 \\ -0.4 & 0.1 & 0.6 & 0.3 \\ 0.7 & -0.2 & 0.4 & -0.5 \\ 0.1 & 0.8 & -0.1 & 0.2 \end{pmatrix}, \quad b_1^{[0:4]} = \begin{pmatrix} 0.1 \\ -0.05 \\ 0.0 \\ 0.15 \end{pmatrix}
$$

<span style="font-size: 14px;">Computing $h_i = x \cdot W_1^{[i]} + b_1^{[i]}$:</span>

* <span style="font-size: 14px;">**Neuron 0:** $(0.5)(0.2) + (-0.3)(0.5) + (0.8)(-0.3) + (-0.1)(0.1) + 0.1 = -0.20$</span>
* <span style="font-size: 14px;">**Neuron 1:** $(0.5)(-0.4) + (-0.3)(0.1) + (0.8)(0.6) + (-0.1)(0.3) - 0.05 = 0.17$</span>
* <span style="font-size: 14px;">**Neuron 2:** $(0.5)(0.7) + (-0.3)(-0.2) + (0.8)(0.4) + (-0.1)(-0.5) + 0.0 = 0.78$</span>
* <span style="font-size: 14px;">**Neuron 3:** $(0.5)(0.1) + (-0.3)(0.8) + (0.8)(-0.1) + (-0.1)(0.2) + 0.15 = -0.14$</span>

<span style="font-size: 14px;">Intermediate: $h^{[0:4]} = [-0.20, 0.17, 0.78, -0.14]$.</span>

<span style="font-size: 14px;">**Step 2 -- Apply GELU.** Using the $\tanh$ approximation on each neuron:</span>

* <span style="font-size: 14px;">**Neuron 0:** $x = -0.20$. $\text{GELU}(-0.20) = 0.5(-0.20)(1 + \tanh(0.7979 \cdot (-0.2004))) = 0.5(-0.20)(0.8413) = -0.0841$</span>
* <span style="font-size: 14px;">**Neuron 1:** $x = 0.17$. $\text{GELU}(0.17) = 0.5(0.17)(1 + \tanh(0.7979 \cdot 0.1702)) = 0.5(0.17)(1.135) = 0.0965$</span>
* <span style="font-size: 14px;">**Neuron 2:** $x = 0.78$. $\text{GELU}(0.78) = 0.5(0.78)(1 + \tanh(0.7979 \cdot 0.8012)) = 0.5(0.78)(1.564) = 0.6101$</span>
* <span style="font-size: 14px;">**Neuron 3:** $x = -0.14$. $\text{GELU}(-0.14) = 0.5(-0.14)(1 + \tanh(0.7979 \cdot (-0.1401))) = 0.5(-0.14)(0.889) = -0.0622$</span>

<span style="font-size: 14px;">After GELU: $a^{[0:4]} = [-0.0841, 0.0965, 0.6101, -0.0622]$.</span>

<span style="font-size: 14px;">Notice how GELU treats negative values differently from ReLU. ReLU would have zeroed out neurons 0 and 3 entirely ($\max(0, -0.20) = 0$, $\max(0, -0.14) = 0$). GELU instead produces small negative outputs, preserving gradient flow.</span>

<span style="font-size: 14px;">**Step 3 -- Contract via $W_2$.** The full 16-dimensional activated vector $a$ is projected back to $d = 4$ by $W_2 \in \mathbb{R}^{4 \times 16}$ plus bias $b_2$. Assuming the full computation yields $\text{FFN}(x) = [0.31, -0.12, 0.45, 0.08]$, the residual addition gives $x + \text{FFN}(x) = [0.81, -0.42, 1.25, -0.02]$.</span>

---

## <span style="font-size: 16px;">Parameter Count</span>

<span style="font-size: 14px;">The FFN dominates parameter count in a Transformer block. For model dimension $d$ with 4x expansion:</span>

* <span style="font-size: 14px;">**$W_1$:** $4d \times d$ parameters</span>
* <span style="font-size: 14px;">**$b_1$:** $4d$ parameters</span>
* <span style="font-size: 14px;">**$W_2$:** $d \times 4d$ parameters</span>
* <span style="font-size: 14px;">**$b_2$:** $d$ parameters</span>

<span style="font-size: 14px;">**Total FFN parameters per block:**</span>

$$
\text{FFN params} = 2 \times 4d \times d + 4d + d = 8d^2 + 5d
$$

<span style="font-size: 14px;">For GPT-2 Small ($d = 768$): $8 \times 768^2 + 5 \times 768 = 4{,}722{,}432$ per block. Compare to attention in the same block: $W_Q, W_K, W_V, W_O \in \mathbb{R}^{d \times d}$ plus biases gives roughly $4d^2 + 4d = 2{,}362{,}368$. The FFN has **twice** the parameters of attention.</span>

<span style="font-size: 14px;">Across all 12 blocks in GPT-2 Small, the FFN contributes $12 \times 4{,}722{,}432 \approx 56.7M$ parameters out of ~124M total. The FFN accounts for about two-thirds of per-block parameters and nearly half of total model parameters.</span>

---

## <span style="font-size: 16px;">Modern Context</span>

<span style="font-size: 14px;">While GPT-2's GELU FFN was standard for years, newer architectures have moved to **SwiGLU** (Shazeer, 2020), now dominant in LLaMA, Mistral, PaLM, and Gemma. The SwiGLU FFN uses a three-matrix gated design:</span>

$$
\text{SwiGLU}(x) = (\text{SiLU}(xW_{\text{gate}}^T) \odot xW_{\text{up}}^T) \cdot W_{\text{down}}^T
$$

<span style="font-size: 14px;">where $\text{SiLU}(x) = x \cdot \sigma(x)$ and $\odot$ is element-wise multiplication. Key differences:</span>

* <span style="font-size: 14px;">**Three matrices instead of two:** $W_{\text{gate}}$, $W_{\text{up}}$, $W_{\text{down}}$ replace $W_1$ and $W_2$. Two parallel projections (one gated, one ungated) replace the single expansion.</span>
* <span style="font-size: 14px;">**Adjusted expansion ratio:** To match parameter count, SwiGLU uses $\frac{8}{3}d$ instead of $4d$. Three matrices of size $\frac{8}{3}d \times d$ yield $3 \times \frac{8}{3}d \times d = 8d^2$, matching GPT-2's $8d^2$.</span>
* <span style="font-size: 14px;">**No biases:** Modern SwiGLU implementations remove bias terms entirely, as biases add negligible capacity at scale.</span>

<span style="font-size: 14px;">The progression: ReLU (2017) to GELU (GPT-2/BERT, 2018-2019) to SwiGLU (LLaMA/PaLM, 2022-2023). Each step improved training stability and performance, though GELU remains correct for GPT-2.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

<span style="font-size: 14px;">**1. Wrong expansion ratio.**</span>

<span style="font-size: 14px;">GPT-2 uses exactly 4x expansion. The intermediate dimension $d_{\text{ff}}$ must be $4 \times d$. For GPT-2 Small with $d = 768$, the intermediate is $3072$, not $1536$ or $2048$.</span>

<span style="font-size: 14px;">**2. Applying GELU after $W_2$ instead of after $W_1$.**</span>

<span style="font-size: 14px;">The activation goes between the two linear layers: linear ($W_1$), GELU, linear ($W_2$). Computing $\text{GELU}(xW_1^TW_2^T)$ -- applying GELU to the final output -- defeats the expand-contract design because the activation no longer operates in the high-dimensional space. It would also distort the residual addition.</span>

<span style="font-size: 14px;">**3. Wrong weight dimensions.**</span>

<span style="font-size: 14px;">$W_1$ has shape $(4d, d)$ and $W_2$ has shape $(d, 4d)$. Swapping these makes $W_1$ contract instead of expand. Remember: $W_1$ maps $d \to 4d$ ($4d$ rows, $d$ columns); $W_2$ maps $4d \to d$ ($d$ rows, $4d$ columns).</span>

<span style="font-size: 14px;">**4. Confusing FFN with attention.**</span>

<span style="font-size: 14px;">Attention is the only cross-position operation. The FFN operates on each position independently. If your implementation produces interactions between positions, something is wrong -- the FFN should operate element-wise on the last dimension of $(B, T, d)$.</span>

<span style="font-size: 14px;">**5. Forgetting biases.**</span>

<span style="font-size: 14px;">GPT-2's FFN includes $b_1$ and $b_2$. Modern architectures like LLaMA remove biases, but for GPT-2 specifically, omitting them is incorrect.</span>

<span style="font-size: 14px;">**6. Using the wrong GELU variant.**</span>

<span style="font-size: 14px;">GPT-2 uses the $\tanh$-based approximation, not exact GELU (`erf`) and not ReLU. The difference is small but matters for reproducibility.</span>

---
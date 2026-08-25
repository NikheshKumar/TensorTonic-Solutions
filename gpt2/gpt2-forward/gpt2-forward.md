# <span style="font-size: 20px;">GPT-2 Forward Pass</span>

<span style="font-size: 14px;">The GPT-2 forward pass transforms a sequence of token IDs into logits over the vocabulary. It chains four stages: token + position embedding, a stack of decoder blocks, a final LayerNorm, and a linear LM head. The output is raw logits with no softmax applied.</span>

<span style="font-size: 14px;">This is the capstone of the GPT-2 architecture. Every component -- token embeddings, positional embeddings, causal multi-head attention, GELU FFN, and LayerNorm -- is assembled into a single autoregressive language model. GPT-2 uses pre-norm (LayerNorm before each sub-layer), differing from the original Transformer's post-norm design.</span>

---

## <span style="font-size: 16px;">What It Is / What It Does</span>

<span style="font-size: 14px;">The forward pass is a four-stage pipeline that converts token IDs to logits:</span>

* <span style="font-size: 14px;">**Stage 1 -- Embedding:** Each token ID maps to a dense vector via a learned embedding table. A learned positional embedding is added, injecting absolute position information.</span>
* <span style="font-size: 14px;">**Stage 2 -- N Decoder Blocks:** The sequence passes through $N$ blocks. Each applies pre-LayerNorm causal MHA followed by pre-LayerNorm GELU FFN, both with residual connections.</span>
* <span style="font-size: 14px;">**Stage 3 -- Final LayerNorm:** Normalizes hidden states after the last block so the LM head receives inputs at a consistent scale.</span>
* <span style="font-size: 14px;">**Stage 4 -- LM Head:** Linear projection from $d$ to $V$ produces logits. No softmax. GPT-2 ties the LM head with the token embedding matrix.</span>

<span style="font-size: 14px;">The input is a tensor of token IDs with shape $(B, T)$ where $B$ is batch size and $T$ is sequence length ($T \leq 1024$ for GPT-2). The output is a logits tensor of shape $(B, T, V)$ where $V$ is vocabulary size (50,257 for GPT-2).</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">**Stage 1 -- Embedding (token + position):**</span>

$$
h^{(0)} = W_e[\text{token\_ids}] + W_p[\text{positions}]
$$

<span style="font-size: 14px;">where $W_e \in \mathbb{R}^{V \times d}$ is the token embedding matrix, $W_p \in \mathbb{R}^{T_{\max} \times d}$ is the positional embedding matrix ($T_{\max} = 1024$), and $\text{positions} = [0, 1, \ldots, T-1]$.</span>

<span style="font-size: 14px;">**Stage 2 -- Each decoder block $\ell$ (for $\ell = 0, 1, \ldots, N-1$):**</span>

<span style="font-size: 14px;">Attention sub-layer with pre-norm and residual:</span>

$$
h^{(\ell+0.5)} = h^{(\ell)} + \text{MHA}(\text{LayerNorm}(h^{(\ell)}))
$$

<span style="font-size: 14px;">FFN sub-layer with pre-norm and residual:</span>

$$
h^{(\ell+1)} = h^{(\ell+0.5)} + \text{FFN}(\text{LayerNorm}(h^{(\ell+0.5)}))
$$

<span style="font-size: 14px;">where LayerNorm is $\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$ with $\epsilon = 10^{-5}$, and each sub-layer has its own $\gamma, \beta \in \mathbb{R}^d$.</span>

<span style="font-size: 14px;">**Stage 3 -- Final LayerNorm:**</span>

$$
h_f = \text{LayerNorm}(h^{(N)}) = \gamma_f \odot \frac{h^{(N)} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta_f
$$

<span style="font-size: 14px;">**Stage 4 -- LM head projection (with weight tying):**</span>

$$
\text{logits} = h_f \cdot W_e^T
$$

<span style="font-size: 14px;">where the LM head reuses the token embedding matrix $W_e \in \mathbb{R}^{V \times d}$. No bias is added. No softmax is applied.</span>

---

## <span style="font-size: 16px;">Stage 1: Embedding (Token + Position)</span>

<span style="font-size: 14px;">GPT-2 uses two separate learned embedding tables summed element-wise. The token embedding $W_e \in \mathbb{R}^{V \times d}$ maps each token ID to a dense vector. The positional embedding $W_p \in \mathbb{R}^{1024 \times d}$ maps each position index to a vector of the same dimension.</span>

$$
h^{(0)}_i = W_e[t_i] + W_p[i], \quad i = 0, 1, \ldots, T-1
$$

<span style="font-size: 14px;">Both embeddings are fully learned (no sinusoidal formulas). Positions are injected once at the input, not inside each block. Every block sees the same fused representation of identity and position.</span>

* <span style="font-size: 14px;">**Simplicity:** No hand-designed frequency formulas. The model learns positional patterns from data.</span>
* <span style="font-size: 14px;">**Additive fusion:** Token and position embeddings live in the same $d$-dimensional space and combine by addition.</span>
* <span style="font-size: 14px;">**Fixed maximum length:** The positional table has exactly 1024 rows. Sequences exceeding this cannot be processed without modifying the embedding.</span>

<span style="font-size: 14px;">The combined embedding $h^{(0)}$ has shape $(B, T, d)$ and enters the decoder block stack.</span>

---

## <span style="font-size: 16px;">Stage 2: Decoder Blocks</span>

<span style="font-size: 14px;">The core of GPT-2 is a stack of $N$ identical decoder blocks applied sequentially. Each block has its own learned parameters but follows the same template. Block $\ell$ transforms $h^{(\ell)}$ into $h^{(\ell+1)}$ through two sub-layers.</span>

<span style="font-size: 14px;">**Sub-layer 1: Pre-norm Causal Multi-Head Attention.** LayerNorm is applied to the input first (pre-norm), then Q, K, V are projected:</span>

$$
Q = \text{LN}(h^{(\ell)}) W_Q, \quad K = \text{LN}(h^{(\ell)}) W_K, \quad V = \text{LN}(h^{(\ell)}) W_V
$$

<span style="font-size: 14px;">where $W_Q, W_K, W_V \in \mathbb{R}^{d \times d}$. Reshaped into $n_h$ heads ($d_h = d / n_h$), causal attention is:</span>

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^T}{\sqrt{d_h}} + M\right) V
$$

<span style="font-size: 14px;">where $M$ is a causal mask setting future positions to $-\infty$. Heads are concatenated, projected through $W_O \in \mathbb{R}^{d \times d}$, and added via residual:</span>

$$
h^{(\ell+0.5)} = h^{(\ell)} + W_O \cdot \text{concat}(\text{head}_1, \ldots, \text{head}_{n_h})
$$

<span style="font-size: 14px;">**Sub-layer 2: Pre-norm GELU FFN.** Another LayerNorm, then a two-layer MLP:</span>

$$
\text{FFN}(x) = \text{GELU}(x W_1 + b_1) W_2 + b_2
$$

<span style="font-size: 14px;">where $W_1 \in \mathbb{R}^{d \times 4d}$ expands by 4x and $W_2 \in \mathbb{R}^{4d \times d}$ projects back. GELU is $\text{GELU}(x) = x \cdot \frac{1}{2}[1 + \text{erf}(x/\sqrt{2})]$. With residual:</span>

$$
h^{(\ell+1)} = h^{(\ell+0.5)} + \text{FFN}(\text{LayerNorm}(h^{(\ell+0.5)}))
$$

<span style="font-size: 14px;">Key architectural details:</span>

* <span style="font-size: 14px;">**Pre-norm ordering:** LayerNorm before each sub-layer, not after. Gradients flow directly through the residual path, stabilizing deep training.</span>
* <span style="font-size: 14px;">**Causal masking:** Token at position $i$ attends only to positions $0, 1, \ldots, i$. This makes GPT-2 autoregressive.</span>
* <span style="font-size: 14px;">**Decoder-only:** No encoder, no cross-attention. Every block has exactly two sub-layers: self-attention and FFN.</span>
* <span style="font-size: 14px;">**Biases everywhere:** Unlike later models (LLaMA, Mistral), GPT-2 includes bias terms in all linear projections and both LayerNorm parameters ($\gamma$ and $\beta$).</span>

---

## <span style="font-size: 16px;">Stage 3: Final LayerNorm</span>

<span style="font-size: 14px;">After the last decoder block, $h^{(N)}$ passes through a final LayerNorm:</span>

$$
h_f = \gamma_f \odot \frac{h^{(N)} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta_f
$$

<span style="font-size: 14px;">where $\mu = \frac{1}{d}\sum_{j=1}^{d} h^{(N)}_j$ is the mean, $\sigma^2 = \frac{1}{d}\sum_{j=1}^{d}(h^{(N)}_j - \mu)^2$ is the variance, $\gamma_f, \beta_f \in \mathbb{R}^d$ are learned parameters, and $\epsilon = 10^{-5}$.</span>

<span style="font-size: 14px;">Why is this necessary?</span>

* <span style="font-size: 14px;">**Scale normalization:** The residual stream accumulates contributions from $N$ blocks. Without normalization, hidden state magnitudes vary wildly, producing unstable logits.</span>
* <span style="font-size: 14px;">**Pre-norm consequence:** In pre-norm, the last sub-layer's output is added directly to the residual stream without normalization. The final LayerNorm ensures the LM head receives well-conditioned inputs.</span>
* <span style="font-size: 14px;">**Training stability:** Normalizing before the LM head prevents gradient explosions from large logit values.</span>

<span style="font-size: 14px;">This final LayerNorm is separate from the per-block norms, with its own $\gamma_f$ and $\beta_f$. Forgetting it is one of the most common implementation errors.</span>

---

## <span style="font-size: 16px;">Stage 4: LM Head</span>

<span style="font-size: 14px;">The LM head projects from hidden dimension $d$ to vocabulary size $V$:</span>

$$
\text{logits} = h_f \cdot W_e^T, \quad W_e \in \mathbb{R}^{V \times d}
$$

<span style="font-size: 14px;">The output logits have shape $(B, T, V)$. No softmax is applied -- the model outputs raw logits.</span>

<span style="font-size: 14px;">**Why no softmax?** Cross-entropy loss applies log-softmax internally. Applying softmax in the model causes double-softmax with near-uniform distributions and near-zero gradients. Raw logits also allow temperature scaling and top-k/top-p filtering.</span>

<span style="font-size: 14px;">**Weight tying.** GPT-2 ties the LM head with the token embedding: $W_{\text{head}} = W_e$.</span>

* <span style="font-size: 14px;">**Parameter efficiency:** For GPT-2 Small ($V = 50257$, $d = 768$), tying saves ~38.6M parameters that would otherwise duplicate the embedding.</span>
* <span style="font-size: 14px;">**Semantic consistency:** Tokens with similar embeddings produce similar logits, enforcing a shared semantic geometry between input and output spaces.</span>
* <span style="font-size: 14px;">**No bias:** The tied projection is a pure matrix multiply with no additive bias term.</span>

<span style="font-size: 14px;">Weight tying is optional. Some implementations use a separate untied LM head for more expressiveness.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">GPT-2 was introduced by Radford et al. in "Language Models are Unsupervised Multitask Learners" (2019). The paper showed that a large language model trained on diverse web text (WebText, ~40GB) could perform downstream tasks without fine-tuning. GPT-2 comes in four sizes:</span>

* <span style="font-size: 14px;">**GPT-2 Small (117M):** $d = 768$, $N = 12$ layers, $n_h = 12$ heads, $d_h = 64$</span>
* <span style="font-size: 14px;">**GPT-2 Medium (345M):** $d = 1024$, $N = 24$ layers, $n_h = 16$ heads, $d_h = 64$</span>
* <span style="font-size: 14px;">**GPT-2 Large (762M):** $d = 1280$, $N = 36$ layers, $n_h = 20$ heads, $d_h = 64$</span>
* <span style="font-size: 14px;">**GPT-2 XL (1.5B):** $d = 1600$, $N = 48$ layers, $n_h = 25$ heads, $d_h = 64$</span>

<span style="font-size: 14px;">All variants share the same architecture; only hyperparameters change. Key decisions distinguishing GPT-2 from the original Transformer:</span>

* <span style="font-size: 14px;">**Pre-norm instead of post-norm:** LayerNorm before each sub-layer improves gradient flow and training stability for deep networks.</span>
* <span style="font-size: 14px;">**Learned positional embeddings:** A learned table with 1024 entries replaces sinusoidal encodings, fixing the maximum sequence length.</span>
* <span style="font-size: 14px;">**GELU activation:** Smooth, non-monotonic activation replacing ReLU, avoiding the "dying neuron" problem.</span>
* <span style="font-size: 14px;">**Byte-Pair Encoding:** 50,257-token vocabulary built with BPE, handling any UTF-8 text without unknown tokens.</span>
* <span style="font-size: 14px;">**Weight tying:** Token embedding and LM head share the same weight matrix, reducing parameter count.</span>

<span style="font-size: 14px;">The head dimension $d_h = 64$ is constant across all sizes. Scaling increases $d$, $N$, and $n_h$ together. The FFN intermediate dimension is always $4d$.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Trace 3 token IDs through a simplified GPT-2 with $d = 4$, $V = 6$, $N = 2$ blocks, $n_h = 2$ heads ($d_h = 2$).</span>

<span style="font-size: 14px;">**Input token IDs:** $[3, 1, 5]$ (sequence length $T = 3$).</span>

<span style="font-size: 14px;">**Stage 1 -- Token + Position Embedding.** Look up from $W_e \in \mathbb{R}^{6 \times 4}$ and $W_p \in \mathbb{R}^{8 \times 4}$, then sum:</span>

* <span style="font-size: 14px;">$h_0^{(0)} = W_e[3] + W_p[0] = [0.5, -0.3, 0.8, 0.2] + [0.1, 0.0, -0.1, 0.1] = [0.6, -0.3, 0.7, 0.3]$</span>
* <span style="font-size: 14px;">$h_1^{(0)} = W_e[1] + W_p[1] = [-0.4, 0.6, 0.1, -0.5] + [0.2, -0.1, 0.0, 0.2] = [-0.2, 0.5, 0.1, -0.3]$</span>
* <span style="font-size: 14px;">$h_2^{(0)} = W_e[5] + W_p[2] = [0.7, 0.2, -0.6, 0.3] + [0.0, 0.1, 0.1, -0.1] = [0.7, 0.3, -0.5, 0.2]$</span>

<span style="font-size: 14px;">**Stage 2 -- Block 0.** Focus on position 0. Pre-norm: $\mu = 0.325$, $\sigma^2 = 0.1519$. With $\gamma = [1,1,1,1]$, $\beta = [0,0,0,0]$: $\hat{x}_0 = [0.705, -1.603, 0.962, -0.064]$.</span>

<span style="font-size: 14px;">After causal MHA (token 0 attends only to itself), suppose attention output is $[0.08, -0.12, 0.10, -0.03]$. Residual: $h_0^{(0.5)} = [0.68, -0.42, 0.80, 0.27]$.</span>

<span style="font-size: 14px;">FFN: normalize, expand $4\times$, GELU, project back. Suppose output $[0.14, 0.05, -0.09, 0.11]$. After residual: $h_0^{(1)} = [0.82, -0.37, 0.71, 0.38]$.</span>

<span style="font-size: 14px;">**Block 1.** Same process, independent parameters. Output: $h_0^{(2)} = [0.91, -0.28, 0.65, 0.44]$.</span>

<span style="font-size: 14px;">**Stage 3 -- Final LayerNorm.** $\mu = 0.43$, $\sigma^2 = 0.1958$, $\sqrt{0.1958 + 10^{-5}} \approx 0.4425$. With unit $\gamma_f$, zero $\beta_f$:</span>

$$
h_f = [1.085, -1.604, 0.497, 0.023]
$$

<span style="font-size: 14px;">**Stage 4 -- LM Head (tied weights).** Dot-product $h_f$ with each row of $W_e$:</span>

* <span style="font-size: 14px;">**Logit 0:** $[0.1, 0.4, -0.2, 0.3] \cdot h_f = 0.109 - 0.642 - 0.099 + 0.007 = -0.625$</span>
* <span style="font-size: 14px;">**Logit 1:** $[-0.4, 0.6, 0.1, -0.5] \cdot h_f = -0.434 - 0.962 + 0.050 - 0.012 = -1.358$</span>
* <span style="font-size: 14px;">**Logit 3:** $[0.5, -0.3, 0.8, 0.2] \cdot h_f = 0.543 + 0.481 + 0.398 + 0.005 = 1.427$</span>

<span style="font-size: 14px;">The highest logit (token 3 at 1.427) is the predicted next token. No softmax. Logit 3 is highest because $h_f$ is most similar to token 3's embedding -- a consequence of weight tying.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

<span style="font-size: 14px;">**1. Forgetting the final LayerNorm.**</span>

<span style="font-size: 14px;">Hidden states enter the LM head with unconstrained magnitude, causing wildly varying logit scales and training instability. Per-block norms only normalize sub-layer inputs, not the final residual stream output.</span>

<span style="font-size: 14px;">**2. Applying softmax after the LM head.**</span>

<span style="font-size: 14px;">The forward pass outputs raw logits. Applying softmax then using `CrossEntropyLoss` (which internally computes log-softmax) produces double-softmax with near-uniform distributions and near-zero gradients.</span>

<span style="font-size: 14px;">**3. Wrong number of decoder blocks.**</span>

<span style="font-size: 14px;">Each variant has a specific block count: 12 (Small), 24 (Medium), 36 (Large), 48 (XL). Wrong $N$ breaks pretrained weight compatibility. Always make $N$ configurable.</span>

<span style="font-size: 14px;">**4. Positional embedding exceeding the maximum sequence length.**</span>

<span style="font-size: 14px;">The positional embedding table has exactly 1024 rows. If $T > 1024$, position indices exceed the table, causing an out-of-bounds error. Unlike RoPE-based models, GPT-2 cannot extrapolate to unseen positions.</span>

<span style="font-size: 14px;">**5. Using post-norm instead of pre-norm.**</span>

<span style="font-size: 14px;">The original Transformer uses post-norm (LayerNorm after residual). GPT-2 uses pre-norm (before each sub-layer). Pre-norm allows gradients to flow directly through the residual path. Swapping to post-norm breaks pretrained weight compatibility.</span>

<span style="font-size: 14px;">**6. Forgetting the causal mask in attention.**</span>

<span style="font-size: 14px;">Each token must only attend to itself and previous tokens. Omitting the causal mask leaks future information, producing a model that cannot generate text autoregressively. The mask must set all entries above the diagonal to $-\infty$ before softmax.</span>
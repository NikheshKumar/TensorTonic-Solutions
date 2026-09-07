# <span style="font-size: 20px;">Dense SwiGLU FFN</span>

<span style="font-size: 14px;">SwiGLU is the feed-forward block used in the dense prefix layer of GLM-4.5 (Zeng et al., 2025). It combines a **Gated Linear Unit** with the **SiLU** (Swish) activation, replacing the classic two-matrix ReLU FFN of the original Transformer.</span>

---

## <span style="font-size: 16px;">What It Computes</span>

<span style="font-size: 14px;">Given input tokens $x \in \mathbb{R}^{n \times d}$ where $d$ is the hidden size, SwiGLU produces:</span>

$$
\text{SwiGLU}(x) = \big(\text{SiLU}(x W_{\text{gate}}) \odot (x W_{\text{up}})\big) W_{\text{down}}
$$

<span style="font-size: 14px;">where:</span>

* <span style="font-size: 14px;">$W_{\text{gate}}, W_{\text{up}} \in \mathbb{R}^{d \times d_{\text{ff}}}$ project up to the intermediate size $d_{\text{ff}}$.</span>
* <span style="font-size: 14px;">$W_{\text{down}} \in \mathbb{R}^{d_{\text{ff}} \times d}$ projects back to the hidden size.</span>
* <span style="font-size: 14px;">$\odot$ is the elementwise product.</span>
* <span style="font-size: 14px;">$\text{SiLU}(z) = z \cdot \sigma(z)$ with $\sigma$ the logistic sigmoid.</span>
* <span style="font-size: 14px;">There are no biases. GLM-4.5, LLaMA, Mistral, and DeepSeek all drop FFN biases.</span>

---

## <span style="font-size: 16px;">The GLU Family</span>

<span style="font-size: 14px;">Gated Linear Units come from Dauphin et al. (2017). The pattern is to split a linear projection in two and let one half gate the other:</span>

$$
\text{GLU}(x, W, V) = \sigma(xW) \odot (xV)
$$

<span style="font-size: 14px;">Shazeer (2020) ran the systematic sweep and showed that swapping the sigmoid for other activations gives a family of variants:</span>

* <span style="font-size: 14px;">**ReGLU:** $\text{ReLU}(xW) \odot (xV)$</span>
* <span style="font-size: 14px;">**GEGLU:** $\text{GELU}(xW) \odot (xV)$</span>
* <span style="font-size: 14px;">**SwiGLU:** $\text{SiLU}(xW) \odot (xV)$, the variant used in PaLM, LLaMA, Mistral, DeepSeek, and GLM-4.5.</span>

<span style="font-size: 14px;">Shazeer's paper measured these on T5 pretraining and downstream tasks. SwiGLU and GEGLU consistently beat the vanilla ReLU FFN on perplexity, with SwiGLU winning narrowly. Shazeer famously closed the paper by saying the improvements come "by divine benevolence" because there is no clean theoretical justification, only empirics.</span>

---

## <span style="font-size: 16px;">SiLU (Swish)</span>

<span style="font-size: 14px;">SiLU is defined as $\text{SiLU}(z) = z \cdot \sigma(z)$, where $\sigma(z) = 1 / (1 + e^{-z})$. It was first proposed by Elfwing et al. (2017) under the name SiLU and independently rediscovered as "Swish" by Ramachandran et al. (2017) via neural architecture search over activation functions.</span>

<span style="font-size: 14px;">Key properties of SiLU:</span>

* <span style="font-size: 14px;">**Smooth everywhere.** Differentiable at zero, unlike ReLU which has a kink. The derivative is $\text{SiLU}'(z) = \sigma(z) + z \sigma(z)(1 - \sigma(z))$.</span>
* <span style="font-size: 14px;">**Non-monotonic.** Has a small negative dip with minimum value $\approx -0.2785$ at $z \approx -1.2785$. Then climbs back to zero as $z \to -\infty$.</span>
* <span style="font-size: 14px;">**Approximately linear for large positive $z$.** Behaves like $z$ when $z \gg 0$ because $\sigma(z) \to 1$. This keeps gradients flowing.</span>
* <span style="font-size: 14px;">**Saturates to zero for large negative $z$.** $\sigma(z) \to 0$ exponentially so $\text{SiLU}(z) \to 0$.</span>
* <span style="font-size: 14px;">**Self-gated.** The factor $\sigma(z)$ acts as a soft gate on the input $z$ itself, hence "self-gated".</span>

<span style="font-size: 14px;">The smooth shape gives better gradients than ReLU near the origin and avoids the "dying ReLU" failure mode where a unit gets stuck at zero output and zero gradient. Compared with GELU, SiLU is cheaper to compute (one sigmoid versus the tanh-based GELU approximation or the exact erf-based GELU) and empirically performs within noise of GELU on language modeling.</span>

---

## <span style="font-size: 16px;">Why GLU Beats a Vanilla FFN</span>

<span style="font-size: 14px;">The original Transformer FFN (Vaswani et al., 2017) is two linear layers around a pointwise nonlinearity:</span>

$$
\text{FFN}(x) = \text{ReLU}(x W_1 + b_1) W_2 + b_2
$$

<span style="font-size: 14px;">This is a strict bottleneck: the activation only sees a single projection of $x$. SwiGLU adds a second up-projection ($W_{\text{up}}$) that carries the raw signal, and lets the SiLU-activated gate ($W_{\text{gate}}$) modulate it elementwise. Conceptually:</span>

* <span style="font-size: 14px;">$x W_{\text{gate}}$ asks "how much should this feature pass through?".</span>
* <span style="font-size: 14px;">$x W_{\text{up}}$ provides the actual feature values.</span>
* <span style="font-size: 14px;">The elementwise product is a soft, learned mask.</span>

<span style="font-size: 14px;">This is more expressive than ReLU because the gate can take any continuous value, not just on/off, and the up-projection is a clean linear path that helps gradient flow.</span>

---

## <span style="font-size: 16px;">Parameter Budget vs Vanilla FFN</span>

<span style="font-size: 14px;">A vanilla FFN with hidden $d$ and intermediate $4d$ has $2 \cdot d \cdot 4d = 8d^2$ parameters (ignoring biases). SwiGLU with three matrices and intermediate $d_{\text{ff}}$ has $3 \cdot d \cdot d_{\text{ff}}$ parameters. To match the parameter count of a $4d$ vanilla FFN, papers like LLaMA shrink $d_{\text{ff}}$ from $4d$ to roughly $\tfrac{8}{3} d$, often rounded to a multiple of a hardware-friendly number like 256 to keep tensor cores happy.</span>

<span style="font-size: 14px;">Real-world settings:</span>

* <span style="font-size: 14px;">LLaMA 2 7B: $d = 4096$, $d_{\text{ff}} = 11008$ ($\approx 2.69 d$).</span>
* <span style="font-size: 14px;">GLM-4.5 dense layer: $d = 4096$, $d_{\text{ff}} = 10944$ ($\approx 2.67 d$, a clean multiple of 256).</span>
* <span style="font-size: 14px;">Mistral 7B: $d = 4096$, $d_{\text{ff}} = 14336$ ($3.5 d$, breaks the rule a bit).</span>
* <span style="font-size: 14px;">PaLM 540B: $d = 18432$, $d_{\text{ff}} = 73728$ ($4 d$ kept as in the vanilla FFN).</span>

<span style="font-size: 14px;">The exact ratio is a kernel-friendliness compromise, not a theoretical constant. The total FLOPs per token for the FFN scale linearly with $d_{\text{ff}}$, so the budget is set by training compute and serving cost, then rounded to hit alignment requirements.</span>

---

## <span style="font-size: 16px;">Role in GLM-4.5</span>

<span style="font-size: 14px;">GLM-4.5 (arxiv 2508.06471, 2025) is a Mixture-of-Experts (MoE) model. Most transformer blocks have their FFN replaced by an MoE block with multiple experts and a top-$k$ router. However, the **first transformer block keeps a dense SwiGLU FFN** instead of an MoE block.</span>

* <span style="font-size: 14px;">This is controlled by the config flag $\texttt{first\_k\_dense\_replace}$ (set to 1 in GLM-4.5, meaning the first 1 layer is kept dense and the remaining layers are MoE).</span>
* <span style="font-size: 14px;">DeepSeek-V3 uses the same pattern (with $\texttt{first\_k\_dense\_replace}$ = 3): the early layers are dense, then MoE kicks in for the bulk of the network.</span>
* <span style="font-size: 14px;">**Why dense first?** Empirically, the first layer carries general low-level routing of token embeddings into a useful subspace. Sparsifying it via MoE hurts because every expert would need to relearn the same generic mapping, wasting capacity. Keeping it dense gives the model a stable foundation before specialist experts take over in deeper layers.</span>
* <span style="font-size: 14px;">**Hidden and intermediate sizes.** GLM-4.5 uses $d_{\text{model}} = 4096$ and intermediate $d_{\text{ff}} = 10944$ for the dense layer's SwiGLU FFN. The intermediate dimension is roughly $2.67 \times$ the hidden dimension, in line with the LLaMA convention.</span>
* <span style="font-size: 14px;">**Implementation note.** This problem implements exactly the math of that first-layer dense FFN. The MoE blocks in deeper layers reuse the same SwiGLU formula inside each expert; only the routing on top is different.</span>
* <span style="font-size: 14px;">**Activation pattern.** Inside an MoE block, only a small number of experts fire per token. The dense first layer fires for every token unconditionally, which means it gets gradient signal from the entire training distribution and learns highly general features.</span>

---

## <span style="font-size: 16px;">Comparison with Vanilla FFN and ReGLU</span>

<span style="font-size: 14px;">Three feed-forward families and how they differ:</span>

* <span style="font-size: 14px;">**Vanilla FFN (Vaswani et al. 2017):** $\text{ReLU}(xW_1 + b_1) W_2 + b_2$. Two matmuls, ReLU in the middle, biases. Used in BERT, GPT-2, original Transformer. Hard-clip negatives.</span>
* <span style="font-size: 14px;">**ReGLU (Shazeer 2020):** $\text{ReLU}(xW_{\text{gate}}) \odot (xW_{\text{up}})$ then projected by $W_{\text{down}}$. Has the gating structure but keeps ReLU's hard zero for negative gates.</span>
* <span style="font-size: 14px;">**SwiGLU (GLM-4.5, LLaMA, PaLM):** swaps ReLU for SiLU. Smooth gate, mild negative dip, the empirical winner.</span>

<span style="font-size: 14px;">All three GLU variants share the same three-matrix structure with no biases, differing only in the activation applied to the gate path.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Let $n=1$, $d=2$, $d_{\text{ff}}=2$:</span>

* <span style="font-size: 14px;">$x = [1.0, -1.0]$</span>
* <span style="font-size: 14px;">$W_{\text{gate}} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$, $W_{\text{up}} = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$, $W_{\text{down}} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$</span>

<span style="font-size: 14px;">Step 1, gate path: $g = x W_{\text{gate}} = [1, -1]$. $\text{SiLU}(g) = [1 \cdot \sigma(1), -1 \cdot \sigma(-1)] \approx [0.7311, -0.2689]$.</span>

<span style="font-size: 14px;">Step 2, up path: $u = x W_{\text{up}} = [-1, 1]$.</span>

<span style="font-size: 14px;">Step 3, gated: $\text{SiLU}(g) \odot u \approx [-0.7311, -0.2689]$.</span>

<span style="font-size: 14px;">Step 4, down: $\approx [-0.7311, -0.2689] \cdot I = [-0.7311, -0.2689]$.</span>

<span style="font-size: 14px;">Notice the second component is negative even though the gate input was negative. That is the non-monotonic SiLU dip leaking through, multiplied by a positive $u$. ReLU would have zeroed it out.</span>

---

## <span style="font-size: 16px;">Implementation Notes</span>

* <span style="font-size: 14px;">**Tensor shapes.** With $x \in \mathbb{R}^{n \times d}$ and $W_{\text{gate}}, W_{\text{up}} \in \mathbb{R}^{d \times d_{\text{ff}}}$, the matmul $x W_{\text{gate}}$ produces $(n, d_{\text{ff}})$. After the elementwise product, $W_{\text{down}} \in \mathbb{R}^{d_{\text{ff}} \times d}$ brings the output back to $(n, d)$.</span>
* <span style="font-size: 14px;">**No bias.** $\texttt{nn.Linear(..., bias=False)}$ in every projection. Saves a small amount of compute and parameters. Modern LLMs find biases unnecessary when LayerNorm or RMSNorm precedes the block, because the norm already centers the activations.</span>
* <span style="font-size: 14px;">**SiLU vs Swish-$\beta$.** Some papers use $\text{Swish}_\beta(z) = z \cdot \sigma(\beta z)$ with a learnable $\beta$. SwiGLU as used in LLaMA and GLM-4.5 fixes $\beta = 1$, i.e. plain SiLU. The learnable $\beta$ buys little in practice and complicates inference kernels.</span>
* <span style="font-size: 14px;">**Fused kernels.** Real implementations fuse the gate and up projections into one matmul against a stacked weight $W_{\text{gate up}} \in \mathbb{R}^{d \times 2 d_{\text{ff}}}$, then split the result along the last dimension. Mathematically identical, faster on GPU because it issues one large GEMM instead of two smaller ones.</span>
* <span style="font-size: 14px;">**Backward pass.** The gradient of $\text{SiLU}(z) \odot u$ w.r.t. $z$ is $\text{SiLU}'(z) \odot u$, and w.r.t. $u$ is $\text{SiLU}(z)$. PyTorch handles this automatically via autograd when you use $\texttt{F.silu}$.</span>
* <span style="font-size: 14px;">**Surrounding block.** In GLM-4.5 the dense block is $x \leftarrow x + \text{SwiGLU}(\text{RMSNorm}(x))$ after the attention sublayer. The Pre-Norm placement keeps the residual stream well-conditioned.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Using GELU instead of SiLU.** GEGLU is a real variant but is not what GLM-4.5 uses. The two activations look similar but are not numerically identical, so values drift quickly and the model deviates from the published checkpoint.</span>
* <span style="font-size: 14px;">**Swapping $W_{\text{gate}}$ and $W_{\text{up}}$.** The result of $\text{SiLU}(x W_{\text{up}}) \odot (x W_{\text{gate}})$ is not equal to $\text{SiLU}(x W_{\text{gate}}) \odot (x W_{\text{up}})$ in general because SiLU is nonlinear. Mixing the two roles silently destroys accuracy.</span>
* <span style="font-size: 14px;">**Forgetting the elementwise product.** Returning $\text{SiLU}(x W_{\text{gate}}) W_{\text{down}}$ alone discards the up path. This collapses SwiGLU into an ordinary FFN with SiLU activation, losing the gating mechanism.</span>
* <span style="font-size: 14px;">**Forgetting the down projection.** The output of $\text{SiLU}(g) \odot u$ has shape $(n, d_{\text{ff}})$ which is larger than the hidden size. Skipping $W_{\text{down}}$ produces a shape mismatch in the residual add of the surrounding block.</span>
* <span style="font-size: 14px;">**Plain sigmoid instead of SiLU.** $\sigma(z)$ alone (without the $z \cdot$ prefactor) caps the gate output to $[0, 1]$. SiLU's range is $[-0.278..., +\infty)$, which is qualitatively different.</span>
* <span style="font-size: 14px;">**Using $+$ instead of $\odot$.** Adding the gate and up paths is a residual connection, not a gate. Loses all multiplicative interactions.</span>
* <span style="font-size: 14px;">**Adding biases.** GLM-4.5 and most modern LLMs use $\texttt{bias=False}$ on these projections. Adding a learnable bias matches no published checkpoint and quietly changes the parameter count.</span>
* <span style="font-size: 14px;">**Wrong intermediate size.** The "$4d$" rule from the original Transformer does not apply. GLM-4.5 uses roughly $2.67 d$ for its dense layer; using $4d$ blows up parameters and breaks weight loading.</span>
* <span style="font-size: 14px;">**Transposing the weights.** Some implementations store $W_{\text{gate}}$ as $(d_{\text{ff}}, d)$ and compute $x @ W_{\text{gate}}.T$. Others store it as $(d, d_{\text{ff}})$ and compute $x @ W_{\text{gate}}$ directly. Mixing the two layouts during weight loading silently produces matrix-shape mismatches that fail loudly, or worse, silently runs the wrong matmul if the dims happen to coincide.</span>
* <span style="font-size: 14px;">**Applying the activation to the up path instead of the gate path.** SwiGLU activates the gate, then multiplies by the un-activated up projection. Applying SiLU to $x W_{\text{up}}$ instead is a different function and matches no published model.</span>

---

# <span style="font-size: 20px;">Full Llama 3 Forward Pass</span>

<span style="font-size: 14px;">The full Llama 3 forward pass transforms a sequence of token IDs into logits over the vocabulary. It chains four stages: token embedding, a stack of Transformer blocks, a final RMSNorm, and a linear LM head. The output is raw logits (no softmax).</span>

<span style="font-size: 14px;">This is the capstone of the Llama 3 architecture. Every component studied in isolation -- RMSNorm, RoPE, Grouped Query Attention, and SwiGLU -- is assembled here into a single coherent model.</span>

---

## <span style="font-size: 16px;">What It Is / What It Does</span>

<span style="font-size: 14px;">The forward pass is a four-stage pipeline that converts discrete token IDs into a continuous logit vector over the entire vocabulary:</span>

* <span style="font-size: 14px;">**Stage 1 -- Token Embedding:** Each token ID is mapped to a dense vector via a learned embedding table. No positional embedding is added here because Llama 3 handles position inside each block via RoPE.</span>
* <span style="font-size: 14px;">**Stage 2 -- N Transformer Blocks:** The embedded sequence passes through $N$ blocks sequentially. Each block applies pre-norm RMSNorm + GQA with RoPE + residual, then pre-norm RMSNorm + SwiGLU FFN + residual.</span>
* <span style="font-size: 14px;">**Stage 3 -- Final RMSNorm:** Stabilizes the hidden states before projection so the LM head receives representations at a consistent scale.</span>
* <span style="font-size: 14px;">**Stage 4 -- LM Head:** A linear projection from hidden dimension to vocabulary size produces logits. No softmax is applied. Embedding and LM head weights are untied.</span>

<span style="font-size: 14px;">The input is a tensor of token IDs with shape $(B, T)$ where $B$ is batch size and $T$ is sequence length. The output is a logits tensor of shape $(B, T, V)$ where $V$ is vocabulary size.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">**Stage 1 -- Embedding lookup:**</span>

$$
h^{(0)} = W_{\text{embed}}[\text{token\_ids}]
$$

<span style="font-size: 14px;">where $W_{\text{embed}} \in \mathbb{R}^{V \times d}$, $V$ is vocabulary size, $d$ is hidden dimension, and $h^{(0)} \in \mathbb{R}^{B \times T \times d}$.</span>

<span style="font-size: 14px;">**Stage 2 -- Each Transformer block $i$ (for $i = 0, 1, \ldots, N-1$):**</span>

<span style="font-size: 14px;">Attention sub-layer with pre-norm and residual:</span>

$$
h^{(i+0.5)} = h^{(i)} + \text{GQA}(\text{RMSNorm}(h^{(i)}, \gamma_{\text{attn}}), \text{RoPE})
$$

<span style="font-size: 14px;">FFN sub-layer with pre-norm and residual:</span>

$$
h^{(i+1)} = h^{(i+0.5)} + \text{SwiGLU}(\text{RMSNorm}(h^{(i+0.5)}, \gamma_{\text{ffn}}))
$$

<span style="font-size: 14px;">**Stage 3 -- Final RMSNorm:**</span>

$$
h_f = \text{RMSNorm}(h^{(N)}) = \frac{h^{(N)} \cdot \gamma_f}{\sqrt{\frac{1}{d} \sum_{j=1}^{d} (h^{(N)}_j)^2 + \epsilon}}
$$

<span style="font-size: 14px;">**Stage 4 -- LM head projection:**</span>

$$
\text{logits} = h_f \cdot W_{\text{head}}^T
$$

<span style="font-size: 14px;">where $W_{\text{head}} \in \mathbb{R}^{V \times d}$ is the LM head weight matrix, separate from $W_{\text{embed}}$. The $\epsilon$ in RMSNorm is $10^{-6}$ throughout.</span>

---

## <span style="font-size: 16px;">Stage 1: Token Embedding</span>

<span style="font-size: 14px;">The embedding layer is a lookup table $W_{\text{embed}} \in \mathbb{R}^{V \times d}$. Given a token ID $t$, the embedding is the $t$-th row of the matrix. For a batch of $B$ sequences each of length $T$, this produces a tensor of shape $(B, T, d)$.</span>

<span style="font-size: 14px;">Critically, Llama 3 does **not** add positional embedding at this stage. Classical Transformers (GPT-2, the original "Attention Is All You Need") add sinusoidal or learned positional embeddings before the first block. Llama 3 instead injects position information inside each block through RoPE, which rotates Q and K during attention. The embedding stage is purely semantic -- tokens map to meaning vectors with no position information.</span>

<span style="font-size: 14px;">Why this design?</span>

* <span style="font-size: 14px;">**RoPE is relative, not absolute:** It encodes relative positions through rotation angles on Q and K. No additive embedding at the input is needed.</span>
* <span style="font-size: 14px;">**Per-layer injection:** Position information is injected fresh at every attention layer, rather than added once and hoping it survives through many layers of transformation.</span>
* <span style="font-size: 14px;">**Separation of concerns:** The embedding layer handles vocabulary mapping; each block handles positional encoding independently.</span>

---

## <span style="font-size: 16px;">Stage 2: Transformer Blocks</span>

<span style="font-size: 14px;">The core of the model is a stack of $N$ blocks applied sequentially. Each block has the same structural template but its own learned parameters. Block $i$ transforms $h^{(i)}$ into $h^{(i+1)}$ through two sub-layers.</span>

<span style="font-size: 14px;">**Sub-layer 1 -- GQA with RoPE:**</span>

<span style="font-size: 14px;">1. **Pre-norm:** $\hat{x} = \text{RMSNorm}(h^{(i)}, \gamma_{\text{attn}})$.</span>

<span style="font-size: 14px;">2. **Project Q, K, V:** Q has $n_q$ heads, K and V share $n_{kv}$ heads (where $n_{kv} < n_q$).</span>

<span style="font-size: 14px;">3. **Apply RoPE:** Rotate Q and K by position-dependent angles using precomputed frequencies.</span>

<span style="font-size: 14px;">4. **Repeat KV heads:** Each KV head is replicated $n_q / n_{kv}$ times so every Q head has a matching pair.</span>

<span style="font-size: 14px;">5. **Attention:** Compute $QK^T / \sqrt{d_h}$, apply causal mask, softmax, then multiply by V.</span>

<span style="font-size: 14px;">6. **Residual:** $h^{(i+0.5)} = h^{(i)} + W_o \cdot \text{concat}(\text{heads})$.</span>

<span style="font-size: 14px;">**Sub-layer 2 -- SwiGLU FFN:**</span>

<span style="font-size: 14px;">1. **Pre-norm:** $\hat{h} = \text{RMSNorm}(h^{(i+0.5)}, \gamma_{\text{ffn}})$.</span>

<span style="font-size: 14px;">2. **SwiGLU:**</span>

$$
\text{SwiGLU}(\hat{h}) = (\text{SiLU}(\hat{h} W_{\text{gate}}^T) \odot \hat{h} W_{\text{up}}^T) W_{\text{down}}^T
$$

<span style="font-size: 14px;">where $\text{SiLU}(x) = x \cdot \sigma(x)$ and $\odot$ is element-wise multiplication.</span>

<span style="font-size: 14px;">3. **Residual:** $h^{(i+1)} = h^{(i+0.5)} + \text{SwiGLU}(\hat{h})$.</span>

<span style="font-size: 14px;">Key architectural details:</span>

* <span style="font-size: 14px;">**Pre-norm (not post-norm):** Llama 3 normalizes before each sub-layer. This improves training stability compared to the original Transformer's post-norm design.</span>
* <span style="font-size: 14px;">**RMSNorm (not LayerNorm):** RMSNorm skips mean-centering and only normalizes by the root mean square. Cheaper to compute and equally effective.</span>
* <span style="font-size: 14px;">**GQA (not MHA or MQA):** Groups Q heads to share KV heads, reducing KV cache memory during inference without significant quality loss.</span>
* <span style="font-size: 14px;">**SwiGLU (not ReLU or GELU):** The three-matrix gated FFN ($W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$) consistently outperforms the standard two-matrix ReLU FFN in language modeling.</span>

---

## <span style="font-size: 16px;">Stage 3: Final RMSNorm</span>

<span style="font-size: 14px;">After the last block, $h^{(N)}$ passes through a final RMSNorm:</span>

$$
h_f = \frac{h^{(N)} \cdot \gamma_f}{\sqrt{\frac{1}{d} \sum_{j=1}^{d} (h^{(N)}_j)^2 + \epsilon}}
$$

<span style="font-size: 14px;">where $\gamma_f \in \mathbb{R}^d$ is a learned scale vector and $\epsilon = 10^{-6}$.</span>

<span style="font-size: 14px;">Why is this necessary?</span>

* <span style="font-size: 14px;">**Scale normalization:** The residual stream accumulates contributions from $N$ blocks. The final RMSNorm ensures consistent scale regardless of depth or input.</span>
* <span style="font-size: 14px;">**Stable logit magnitudes:** Without normalization, hidden state norms vary wildly, making logits unstable, training volatile, and sampling unreliable.</span>
* <span style="font-size: 14px;">**Universal practice:** All modern LLMs (Llama, Mistral, Qwen, Gemma) include a final norm before the LM head. Omitting it degrades training stability and generation quality.</span>

<span style="font-size: 14px;">This final RMSNorm is separate from the pre-norm layers inside each block. It has its own learned $\gamma_f$ parameter vector.</span>

---

## <span style="font-size: 16px;">Stage 4: LM Head</span>

<span style="font-size: 14px;">The LM head is a single linear layer projecting from $d$ to $V$:</span>

$$
\text{logits} = h_f \cdot W_{\text{head}}^T, \quad W_{\text{head}} \in \mathbb{R}^{V \times d}
$$

<span style="font-size: 14px;">The output logits have shape $(B, T, V)$. No softmax is applied -- the model outputs raw logits.</span>

<span style="font-size: 14px;">**Why no softmax?** Cross-entropy loss applies log-softmax internally for numerical stability. Applying softmax in the model then using cross-entropy produces double-softmax, which is mathematically wrong and causes near-zero gradients. Raw logits also allow temperature scaling, top-k, and top-p sampling during inference.</span>

<span style="font-size: 14px;">**Why untied weights?** Some architectures (e.g., GPT-2) tie the embedding and LM head ($W_{\text{head}} = W_{\text{embed}}$). Llama 3 uses separate matrices because:</span>

* <span style="font-size: 14px;">**Different roles:** The embedding maps tokens into a space for contextual processing. The LM head maps hidden states into a space for next-token prediction. These are different functions.</span>
* <span style="font-size: 14px;">**More capacity:** Untying adds $V \times d$ parameters but gives more expressiveness. For large models this cost is marginal.</span>
* <span style="font-size: 14px;">**No gradient conflict:** With tied weights, gradients from the loss and from the embedding lookup compete on the same matrix. Untying eliminates this.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">Llama 3 is a family of decoder-only Transformers released by Meta in 2024. The architecture is deliberately simple -- it uses well-established components (RoPE, GQA, SwiGLU, RMSNorm) without novel architectural tricks, focusing instead on scaling data quality and compute.</span>

<span style="font-size: 14px;">The family comes in three sizes:</span>

* <span style="font-size: 14px;">**8B:** $d = 4096$, $N = 32$, 32 Q heads, 8 KV heads, intermediate 14336, vocab 128,256.</span>
* <span style="font-size: 14px;">**70B:** $d = 8192$, $N = 80$, 64 Q heads, 8 KV heads, intermediate 28672, vocab 128,256.</span>
* <span style="font-size: 14px;">**405B:** $d = 16384$, $N = 126$, 128 Q heads, 8 KV heads, intermediate 53248, vocab 128,256.</span>

<span style="font-size: 14px;">Key decisions consistent across all sizes:</span>

* <span style="font-size: 14px;">**RoPE** with base frequency 500,000 (up from Llama 2's 10,000) to support 128K context.</span>
* <span style="font-size: 14px;">**GQA** with 8 KV heads at every scale. Even the 8B model uses GQA, trading slight quality for significant KV cache savings.</span>
* <span style="font-size: 14px;">**SwiGLU** with intermediate dimension roughly $\frac{8}{3} d$ (rounded to the nearest multiple of 1024).</span>
* <span style="font-size: 14px;">**RMSNorm** with $\epsilon = 10^{-6}$ at every normalization point.</span>
* <span style="font-size: 14px;">**Untied weights** for embedding and LM head. The 128,256-token BPE vocabulary is much larger than Llama 2's 32K, improving efficiency for non-English text and code.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Trace 3 token IDs through a simplified model with $d = 4$, $V = 8$, $N = 2$ blocks, 2 Q heads, 1 KV head, head dim $d_h = 2$.</span>

<span style="font-size: 14px;">**Input token IDs:** $[2, 5, 7]$ (batch $B = 1$, length $T = 3$).</span>

<span style="font-size: 14px;">**Stage 1 -- Embedding lookup.** Look up rows 2, 5, 7 from $W_{\text{embed}} \in \mathbb{R}^{8 \times 4}$:</span>

* <span style="font-size: 14px;">**Token 2:** $h_0^{(0)} = [0.3, -0.5, 0.8, 0.1]$</span>
* <span style="font-size: 14px;">**Token 5:** $h_1^{(0)} = [-0.2, 0.7, 0.4, -0.6]$</span>
* <span style="font-size: 14px;">**Token 7:** $h_2^{(0)} = [0.9, 0.1, -0.3, 0.5]$</span>

<span style="font-size: 14px;">No positional embedding added. These raw semantic vectors enter the block stack.</span>

<span style="font-size: 14px;">**Stage 2 -- Block 0.** Focus on token 2:</span>

<span style="font-size: 14px;">Pre-norm (attention): $\text{RMS} = \sqrt{(0.09 + 0.25 + 0.64 + 0.01)/4} = \sqrt{0.2475} \approx 0.4975$. With $\gamma = [1,1,1,1]$: $\hat{x} = [0.603, -1.005, 1.608, 0.201]$.</span>

<span style="font-size: 14px;">After GQA with RoPE (project Q/K/V, apply rotations, repeat KV head, causal attention, output projection), suppose the attention output is $[0.15, -0.08, 0.22, -0.05]$. Residual: $h^{(0.5)} = [0.45, -0.58, 1.02, 0.05]$.</span>

<span style="font-size: 14px;">After pre-norm + SwiGLU + residual, suppose $h^{(1)} = [0.56, -0.51, 0.88, 0.14]$.</span>

<span style="font-size: 14px;">**Stage 2 -- Block 1.** Same process. After block 1: $h^{(2)} = [0.72, -0.39, 0.95, 0.21]$.</span>

<span style="font-size: 14px;">**Stage 3 -- Final RMSNorm.** For $h^{(2)} = [0.72, -0.39, 0.95, 0.21]$:</span>

$$
\text{RMS} = \sqrt{\frac{0.5184 + 0.1521 + 0.9025 + 0.0441}{4}} = \sqrt{0.4043} \approx 0.6358
$$

<span style="font-size: 14px;">With $\gamma_f = [1,1,1,1]$: $h_f = [1.133, -0.614, 1.494, 0.330]$.</span>

<span style="font-size: 14px;">**Stage 4 -- LM Head.** Project through $W_{\text{head}} \in \mathbb{R}^{8 \times 4}$ (untied from embedding):</span>

* <span style="font-size: 14px;">**Logit 0:** $[0.2, 0.1, -0.3, 0.5] \cdot h_f = 0.227 - 0.061 - 0.448 + 0.165 = -0.118$</span>
* <span style="font-size: 14px;">**Logit 1:** $[0.4, -0.2, 0.6, 0.1] \cdot h_f = 0.453 + 0.123 + 0.896 + 0.033 = 1.505$</span>
* <span style="font-size: 14px;">**Logit 2:** $[-0.1, 0.8, 0.2, -0.4] \cdot h_f = -0.113 - 0.491 + 0.299 - 0.132 = -0.437$</span>

<span style="font-size: 14px;">Continuing for all 8 entries produces the full logit vector. The highest logit (1.505 at position 1) is the model's prediction. No softmax applied.</span>

<span style="font-size: 14px;">**Key observations:** The embedding contributes no positional information -- only blocks (via RoPE) know token order. Each block's pre-norm prevents activation explosion. The final RMSNorm ensures consistent magnitude before the LM head. The LM head is independent of the embedding matrix.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

<span style="font-size: 14px;">**1. Adding positional embeddings at the embedding stage.**</span>

<span style="font-size: 14px;">Llama 3 uses RoPE inside each block's attention, not an additive positional embedding at the input. Adding a learned or sinusoidal positional embedding to $h^{(0)}$ would double-count position information and break the architecture.</span>

<span style="font-size: 14px;">**2. Tying embedding and LM head weights.**</span>

<span style="font-size: 14px;">Weight tying ($W_{\text{head}} = W_{\text{embed}}$) is incorrect for Llama 3. The two matrices are independently parameterized. Tying forces one matrix to serve two different functions and creates gradient conflicts.</span>

<span style="font-size: 14px;">**3. Applying softmax after the LM head.**</span>

<span style="font-size: 14px;">The model outputs raw logits. Applying softmax then using cross-entropy loss (which applies log-softmax internally) produces double-softmax with near-zero gradients. Softmax belongs outside the model, applied only during inference for sampling.</span>

<span style="font-size: 14px;">**4. Forgetting the final RMSNorm.**</span>

<span style="font-size: 14px;">Without the final RMSNorm, hidden states enter the LM head with unconstrained magnitude, causing wildly varying logits, training instability, and unreliable generation.</span>

<span style="font-size: 14px;">**5. Using the wrong epsilon value.**</span>

<span style="font-size: 14px;">Llama 3 uses $\epsilon = 10^{-6}$ for all RMSNorm layers. Using $10^{-5}$ (PyTorch LayerNorm default) or $10^{-8}$ produces different numerical results and breaks reproducibility.</span>

---
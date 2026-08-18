# <span style="font-size: 20px;">Multi-head Latent Attention (MLA)</span>

<span style="font-size: 14px;">Multi-head Latent Attention (MLA) is the attention mechanism at the heart of DeepSeek V2 and V3. It replaces standard Multi-Head Attention (MHA) by compressing key-value representations into a low-rank latent space, dramatically reducing the KV cache footprint during inference while preserving full expressiveness. MLA decouples positional encoding (RoPE) from the compressed representation, ensuring position information does not interfere with cache compression.</span>

<span style="font-size: 14px;">This problem integrates three building blocks -- KV compression, KV reconstruction, and decoupled RoPE -- into a complete attention layer with causal masking, output projection, and a residual connection.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">MLA is an attention mechanism that achieves the KV cache efficiency of Multi-Query Attention (MQA) or Grouped-Query Attention (GQA) while retaining the representational capacity of full Multi-Head Attention. The core idea is to never store per-head keys and values in the cache. Instead, MLA stores a single compressed vector $c_{kv}$ per token, and reconstructs per-head keys and values on the fly during attention computation.</span>

<span style="font-size: 14px;">The mechanism has three distinctive design choices:</span>

* <span style="font-size: 14px;">**Low-rank KV compression:** Instead of projecting input $x$ directly into $n_h$ separate key-value head pairs (each of dimension $d_h$), MLA first projects $x$ into a compact latent vector $c_{kv}$ of dimension $d_c$, where $d_c \ll n_h \cdot d_h$. This latent vector is what gets cached.</span>
* <span style="font-size: 14px;">**Up-projection for reconstruction:** Per-head keys ($K_{\text{nope}}$) and values ($V$) are reconstructed from $c_{kv}$ via learned up-projection matrices. The subscript "nope" stands for "no positional encoding" -- these key components carry content information but no position information.</span>
* <span style="font-size: 14px;">**Decoupled RoPE:** Rotary Position Embeddings are applied to separate, small projections ($q_{\text{rope}}$ and $k_{\text{rope}}$) that are concatenated onto the content-based Q and K. This keeps RoPE out of the compressed representation, so the cached $c_{kv}$ remains position-independent.</span>

<span style="font-size: 14px;">After reconstruction and positional encoding, the final Q and K are formed by concatenation, and standard scaled dot-product attention with a causal mask is applied. The output passes through a linear projection and is added back to the input as a residual.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">**Step 1 -- KV Compression:** Project the input into a low-dimensional latent space.</span>

$$
c_{kv} = x \cdot W_{dkv}^T
$$

<span style="font-size: 14px;">where $x \in \mathbb{R}^{B \times T \times d}$, $W_{dkv} \in \mathbb{R}^{d_c \times d}$, and $c_{kv} \in \mathbb{R}^{B \times T \times d_c}$. This is the only thing stored in the KV cache per token.</span>

<span style="font-size: 14px;">**Step 2 -- KV Reconstruction:** Recover per-head keys and values from the latent.</span>

$$
K_{\text{nope}} = \text{reshape}(c_{kv} \cdot W_{uk}^T, \; [B, T, n_h, d_h])
$$

$$
V = \text{reshape}(c_{kv} \cdot W_{uv}^T, \; [B, T, n_h, d_h])
$$

<span style="font-size: 14px;">where $W_{uk}, W_{uv} \in \mathbb{R}^{(n_h \cdot d_h) \times d_c}$. The up-projections expand the latent back to full multi-head dimension.</span>

<span style="font-size: 14px;">**Step 3 -- Query and RoPE projections:**</span>

$$
Q_{\text{nope}} = \text{reshape}(x \cdot W_q^T, \; [B, T, n_h, d_h])
$$

$$
q_{\text{rope}} = \text{RoPE}\bigl(\text{reshape}(x \cdot W_{qr}^T, \; [B, T, n_h, d_{\text{rope}}])\bigr)
$$

$$
k_{\text{rope}} = \text{RoPE}\bigl(\text{reshape}(x \cdot W_{kr}^T, \; [B, T, 1, d_{\text{rope}}])\bigr)
$$

<span style="font-size: 14px;">where $W_q \in \mathbb{R}^{(n_h \cdot d_h) \times d}$, $W_{qr} \in \mathbb{R}^{(n_h \cdot d_{\text{rope}}) \times d}$, $W_{kr} \in \mathbb{R}^{d_{\text{rope}} \times d}$. Note that $k_{\text{rope}}$ is shared across all heads (only 1 head dimension), while $q_{\text{rope}}$ is per-head.</span>

<span style="font-size: 14px;">**Step 4 -- Concatenation:**</span>

$$
Q = [Q_{\text{nope}} \,||\, q_{\text{rope}}] \in \mathbb{R}^{B \times T \times n_h \times (d_h + d_{\text{rope}})}
$$

$$
K = [K_{\text{nope}} \,||\, k_{\text{rope}}] \in \mathbb{R}^{B \times T \times n_h \times (d_h + d_{\text{rope}})}
$$

<span style="font-size: 14px;">The $k_{\text{rope}}$ component is broadcast across all $n_h$ heads before concatenation.</span>

<span style="font-size: 14px;">**Step 5 -- Scaled dot-product attention with causal mask:**</span>

$$
\text{scores} = \frac{Q \cdot K^T}{\sqrt{d_h + d_{\text{rope}}}} + M_{\text{causal}}
$$

$$
\text{Attn} = \text{softmax}(\text{scores}) \cdot V
$$

<span style="font-size: 14px;">where $M_{\text{causal}}$ is a mask with $0$ on and below the diagonal and $-\infty$ above. Attention is computed over the full Q and K (dimension $d_h + d_{\text{rope}}$), but the weighted sum is over V (dimension $d_h$).</span>

<span style="font-size: 14px;">**Step 6 -- Output projection with residual:**</span>

$$
\text{output} = \text{reshape}(\text{Attn}, \; [B, T, n_h \cdot d_h]) \cdot W_o^T + x
$$

<span style="font-size: 14px;">where $W_o \in \mathbb{R}^{d \times (n_h \cdot d_h)}$ projects the concatenated head outputs back to model dimension $d$, and $x$ is added as a residual connection.</span>

---

## <span style="font-size: 16px;">Why MLA Over MHA, GQA, and MQA</span>

<span style="font-size: 14px;">The fundamental bottleneck in Transformer inference is the KV cache, which grows linearly with sequence length. Different attention variants trade off cache size against model quality:</span>

* <span style="font-size: 14px;">**MHA (Multi-Head Attention):** Each head has its own K and V. Cache per token: $2 \times n_h \times d_h$. For DeepSeek V3 (128 heads, $d_h = 128$): $2 \times 128 \times 128 = 32{,}768$ floats per token per layer.</span>
* <span style="font-size: 14px;">**GQA (Grouped-Query Attention):** Groups of heads share K and V. KV cache per token: $2 \times n_g \times d_h$ where $n_g$ is the number of groups. With 8 groups: $2 \times 8 \times 128 = 2{,}048$ floats. Reduces cache 16x but loses per-head specialization in K and V.</span>
* <span style="font-size: 14px;">**MQA (Multi-Query Attention):** All heads share a single K and V. KV cache per token: $2 \times d_h = 256$ floats. Minimal cache but severe representational bottleneck -- every head must attend using the same key and value.</span>
* <span style="font-size: 14px;">**MLA (Multi-head Latent Attention):** Caches only $c_{kv}$ of dimension $d_c$. KV cache per token: $d_c$ floats. With $d_c = 512$, that is 512 floats -- comparable to MQA (256 floats) but without the representational loss because each head reconstructs its own K and V from the shared latent.</span>

<span style="font-size: 14px;">The key insight is that MLA decouples cache size from the number of heads. MHA/GQA/MQA all cache actual key-value vectors, so cache size scales with the number of distinct K/V sets. MLA caches a compressed representation and pays a small compute cost to reconstruct per-head K and V on the fly. During autoregressive inference, the latent $c_{kv}$ is cached instead of full K/V, yielding massive memory savings while matching full MHA quality.</span>

---

## <span style="font-size: 16px;">The Full Pipeline Step by Step</span>

<span style="font-size: 14px;">Tracing through MLA for a single forward pass with input $x$ of shape $(B, T, d)$:</span>

<span style="font-size: 14px;">**1. KV Compression.** Project input to latent space: $c_{kv} = x \cdot W_{dkv}^T$, producing shape $(B, T, d_c)$. No activation function. The latent captures compressed key-value information across all heads.</span>

<span style="font-size: 14px;">**2. Key Reconstruction.** Up-project and reshape: $c_{kv} \cdot W_{uk}^T$ gives $(B, T, n_h \cdot d_h)$, reshaped to $(B, T, n_h, d_h)$. These content keys $K_{\text{nope}}$ encode what each token contains but not where it is.</span>

<span style="font-size: 14px;">**3. Value Reconstruction.** Similarly, $c_{kv} \cdot W_{uv}^T$ is reshaped to $(B, T, n_h, d_h)$ to produce per-head values $V$. Values receive no positional encoding in MLA.</span>

<span style="font-size: 14px;">**4. Query Projection.** The input $x$ (not the latent) is projected: $x \cdot W_q^T$ reshaped to $(B, T, n_h, d_h)$ gives $Q_{\text{nope}}$. Queries come from $x$ directly because they are never cached.</span>

<span style="font-size: 14px;">**5. RoPE Projections.** Two separate projections create the position-carrying components. $q_{\text{rope}} = \text{RoPE}(\text{reshape}(x \cdot W_{qr}^T))$ has shape $(B, T, n_h, d_{\text{rope}})$ (per-head). $k_{\text{rope}} = \text{RoPE}(\text{reshape}(x \cdot W_{kr}^T))$ has shape $(B, T, 1, d_{\text{rope}})$ (shared across heads, since position is the same for all heads at a given token).</span>

<span style="font-size: 14px;">**6. Concatenation.** $Q = [Q_{\text{nope}} \,||\, q_{\text{rope}}]$ and $K = [K_{\text{nope}} \,||\, k_{\text{rope}}]$ along the last dimension, each giving shape $(B, T, n_h, d_h + d_{\text{rope}})$. $k_{\text{rope}}$ is broadcast across all heads before concatenation. The order matters: nope first, rope second.</span>

<span style="font-size: 14px;">**7. Attention Scores.** $Q \cdot K^T / \sqrt{d_h + d_{\text{rope}}}$ produces scores of shape $(B, n_h, T, T)$. The scaling factor uses the total key dimension. A causal mask sets future positions (query index < key index) to $-\infty$.</span>

<span style="font-size: 14px;">**8. Weighted Sum.** Softmax along the key dimension, then multiply by $V$ of shape $(B, n_h, T, d_h)$. The output has shape $(B, n_h, T, d_h)$ -- the value dimension is $d_h$, not $d_h + d_{\text{rope}}$.</span>

<span style="font-size: 14px;">**9. Output Projection and Residual.** Reshape from $(B, n_h, T, d_h)$ to $(B, T, n_h \cdot d_h)$, project by $W_o^T$ to $(B, T, d)$, and add the original input $x$ as a residual.</span>

---

## <span style="font-size: 16px;">Paper Context: DeepSeek V3 Specifics</span>

<span style="font-size: 14px;">In the DeepSeek V3 architecture, MLA is used with the following hyperparameters:</span>

* <span style="font-size: 14px;">**Number of attention heads ($n_h$):** 128. Much larger than typical models (LLaMA 70B uses 64). MLA makes this feasible because cache cost is independent of head count.</span>
* <span style="font-size: 14px;">**Per-head dimension ($d_h$):** 128.</span>
* <span style="font-size: 14px;">**KV compression dimension ($d_c$):** 512. This is 32x smaller than the full KV representation ($128 \times 128 = 16{,}384$ per K or V).</span>
* <span style="font-size: 14px;">**RoPE dimension ($d_{\text{rope}}$):** 64. Each query and key has 64 dimensions dedicated to positional encoding, appended to the content dimensions.</span>
* <span style="font-size: 14px;">**Total Q/K dimension after concatenation:** $d_h + d_{\text{rope}} = 128 + 64 = 192$ per head.</span>
* <span style="font-size: 14px;">**Model dimension ($d$):** 7,168 for DeepSeek V3 (the 671B parameter model).</span>

<span style="font-size: 14px;">The compression ratio is striking. Standard MHA caches $2 \times 128 \times 128 = 32{,}768$ values per token per layer. MLA caches $d_c + d_{\text{rope}} = 512 + 64 = 576$ values -- a 57x reduction.</span>

<span style="font-size: 14px;">DeepSeek V3 also applies RMSNorm to $c_{kv}$ and the query latent before up-projection (an implementation detail omitted from this simplified problem). This stabilizes latent magnitudes before reconstruction.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider a tiny MLA with $B=1$, $T=3$, $d=6$, $n_h=2$, $d_h=2$, $d_c=3$, $d_{\text{rope}}=2$.</span>

<span style="font-size: 14px;">**Input:** $x$ has shape $(1, 3, 6)$. Take the first token's embedding as $x_0 = [1.0, 0.5, -0.3, 0.8, -0.1, 0.4]$.</span>

<span style="font-size: 14px;">**Step 1 -- KV compression.** $c_{kv} = x \cdot W_{dkv}^T$ maps $(1,3,6) \to (1,3,3)$. Each token's 6-dim vector is compressed to 3 dimensions. Suppose $c_{kv,0} = [0.7, -0.2, 0.5]$.</span>

<span style="font-size: 14px;">**Step 2 -- Key reconstruction.** $c_{kv} \cdot W_{uk}^T$ maps $(1,3,3) \to (1,3,4)$ since $n_h \cdot d_h = 2 \times 2 = 4$. Reshape to $(1,3,2,2)$. Suppose head 0 gets $K_{\text{nope},0} = [0.3, -0.1]$ and head 1 gets $K_{\text{nope},1} = [0.6, 0.2]$ for token 0.</span>

<span style="font-size: 14px;">**Step 3 -- Value reconstruction.** Same shapes: $(1,3,3) \to (1,3,4) \to (1,3,2,2)$. Suppose head 0 gets $V_0 = [0.4, 0.1]$ for token 0.</span>

<span style="font-size: 14px;">**Step 4 -- Query projection.** $x \cdot W_q^T$ maps $(1,3,6) \to (1,3,4) \to (1,3,2,2)$. Suppose head 0 gets $Q_{\text{nope},0} = [0.5, -0.3]$ for token 0.</span>

<span style="font-size: 14px;">**Step 5 -- RoPE projections.** Query RoPE: $x \cdot W_{qr}^T$ maps $(1,3,6) \to (1,3,4) \to (1,3,2,2)$. Key RoPE: $x \cdot W_{kr}^T$ maps $(1,3,6) \to (1,3,2) \to (1,3,1,2)$. For position 0, $\cos\theta = 1$, $\sin\theta = 0$, so vectors are unchanged. For position 1, the rotation mixes pairs: $[r_1 \cos\theta - r_2 \sin\theta, \; r_1 \sin\theta + r_2 \cos\theta]$.</span>

<span style="font-size: 14px;">**Step 6 -- Concatenation.** $Q_0 = [0.5, -0.3 \,||\, q_r, q_r'] \in \mathbb{R}^{4}$ per head. $K_0 = [0.3, -0.1 \,||\, k_r, k_r'] \in \mathbb{R}^{4}$ per head. Total per-head Q/K dimension is $d_h + d_{\text{rope}} = 2 + 2 = 4$.</span>

<span style="font-size: 14px;">**Step 7 -- Attention.** Scores: $Q \cdot K^T / \sqrt{4} = Q \cdot K^T / 2$. For token 0 (first position), the causal mask allows attending only to itself. After softmax over one valid position, the weight is 1.0. The output for head 0, token 0 is just $V_0 = [0.4, 0.1]$.</span>

<span style="font-size: 14px;">**Step 8 -- Output.** Concatenate heads: $[0.4, 0.1, v_{h1,0}, v_{h1,1}] \in \mathbb{R}^4$. Project by $W_o^T$ to get $(1, 3, 6)$. Add residual $x$ to get the final output.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Wrong concatenation order.** Q must be $[Q_{\text{nope}} \,||\, q_{\text{rope}}]$ and K must be $[K_{\text{nope}} \,||\, k_{\text{rope}}]$, with content first and position second. Swapping the order means $Q \cdot K^T$ computes cross-terms incorrectly -- position dimensions in Q align with content dimensions in K, producing nonsensical scores.</span>
* <span style="font-size: 14px;">**Applying RoPE to the wrong components.** RoPE must only be applied to $q_{\text{rope}}$ and $k_{\text{rope}}$, never to $Q_{\text{nope}}$ or $K_{\text{nope}}$. The purpose of decoupling is to keep the compressed KV cache position-independent. If RoPE touches $K_{\text{nope}}$, the cached $c_{kv}$ would need to be position-aware, defeating compression and breaking incremental decoding.</span>
* <span style="font-size: 14px;">**Causal mask direction.** The mask must set future positions (where query position < key position) to $-\infty$, not past positions. A common error is masking the lower triangle instead of the upper triangle. After softmax, $-\infty$ becomes 0, preventing information flow from future tokens to past tokens.</span>
* <span style="font-size: 14px;">**Dimension mismatch after concatenation.** Q and K have dimension $d_h + d_{\text{rope}}$ after concatenation, but V remains at dimension $d_h$. The scaling factor in attention should be $\sqrt{d_h + d_{\text{rope}}}$, not $\sqrt{d_h}$. The output projection $W_o$ maps from $n_h \cdot d_h$ (not $n_h \cdot (d_h + d_{\text{rope}})$) because the attention-weighted sum is over V.</span>
* <span style="font-size: 14px;">**Forgetting to broadcast $k_{\text{rope}}$.** The RoPE key has shape $(B, T, 1, d_{\text{rope}})$ since it is shared across heads. Before concatenating with $K_{\text{nope}}$ of shape $(B, T, n_h, d_h)$, it must be expanded to $(B, T, n_h, d_{\text{rope}})$. Omitting this causes a shape error or silent broadcasting along the wrong axis.</span>
* <span style="font-size: 14px;">**Confusing cache contents.** During inference, only $c_{kv}$ (and optionally $k_{\text{rope}}$) should be cached per token. Caching the reconstructed $K_{\text{nope}}$ and $V$ defeats the purpose of MLA and uses the same memory as standard MHA.</span>
* <span style="font-size: 14px;">**Residual connection placement.** The residual adds the original input $x$ after the output projection, not before. The correct form is $x + \text{reshape}(\text{Attn}) \cdot W_o^T$, applied after the projection reduces back to dimension $d$.</span>
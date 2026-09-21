# <span style="font-size: 20px;">GQA Attention with QK-Norm and Partial RoPE</span>

<span style="font-size: 14px;">This problem implements the attention forward pass used in every layer of GLM-4.5 (Zeng et al., 2025). The block fuses three ingredients that the paper combines for stable, efficient long-context attention: **Grouped-Query Attention** (Ainslie et al., 2023) for KV cache savings, **QK-Norm** (Henry et al., 2020; Dehghani et al., 2023) for logit stability, and **partial Rotary Position Embedding** (Su et al., 2021; GPT-NeoX) for positional encoding on a subset of channels. There is no sliding window and no attention sink: every query attends causally to all previous positions under a single softmax.</span>

---

## <span style="font-size: 16px;">What GLM-4.5 Does in the Attention Block</span>

<span style="font-size: 14px;">In the released configuration (96 attention heads, 8 KV heads, head_dim = 128, partial_rotary_factor = 0.5, attention_bias = true), one forward pass through the attention sub-layer of a GLM-4.5 layer performs the following in order:</span>

* <span style="font-size: 14px;">**Project**: a single fused QKV linear (with bias) takes the post-norm hidden state $x \in \mathbb{R}^{n \times d_{\text{model}}}$ to a packed tensor of dimension $(H + 2 H_k) D$, where $H$ is the number of query heads, $H_k$ the number of KV heads, and $D$ the per-head dimension.</span>
* <span style="font-size: 14px;">**Reshape and split**: the packed output is sliced into $Q \in \mathbb{R}^{n \times H \times D}$, $K \in \mathbb{R}^{n \times H_k \times D}$, $V \in \mathbb{R}^{n \times H_k \times D}$.</span>
* <span style="font-size: 14px;">**QK-Norm**: RMSNorm is applied independently to $Q$ and $K$ along their last axis with per-channel gains $\gamma_q, \gamma_k \in \mathbb{R}^D$. Note the gains are shared across heads and across positions.</span>
* <span style="font-size: 14px;">**Partial RoPE**: the first $r$ channels of each Q and K head get rotated by the precomputed $(\cos, \sin)$ tables; the last $D - r$ channels pass through untouched. In GLM-4.5, $r = D/2 = 64$.</span>
* <span style="font-size: 14px;">**GQA expansion**: each KV head is broadcast to $g = H / H_k$ consecutive query heads so $Q K^T$ is shape $(H, n, n)$. In GLM-4.5, $g = 96 / 8 = 12$.</span>
* <span style="font-size: 14px;">**Causal scaled dot-product**: $\text{softmax}(QK^T / \sqrt{D} + M)$ where $M$ is $-\infty$ above the diagonal.</span>
* <span style="font-size: 14px;">**Output projection**: concatenate heads and apply a final linear $W_o$ (with bias here, per the problem signature).</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">Let $x \in \mathbb{R}^{n \times d_{\text{model}}}$ be the input sequence of $n$ tokens. The fused QKV projection is:</span>

$$
[Q \,|\, K \,|\, V] = x W_{qkv} + b_{qkv}, \quad W_{qkv} \in \mathbb{R}^{d_{\text{model}} \times (H + 2 H_k) D}
$$

<span style="font-size: 14px;">After reshaping into heads, RMSNorm on $Q$ and $K$ is:</span>

$$
\text{RMSNorm}(t; \gamma) = \frac{t}{\sqrt{\frac{1}{D} \sum_{j=1}^{D} t_j^2 + \epsilon}} \cdot \gamma
$$

<span style="font-size: 14px;">applied independently to each $(n, h)$ slot. The rotate-half RoPE on a single Q head, applied to its first $r$ channels, is:</span>

$$
q^{\text{rot}}_i = q^{\text{rot}}_i \odot \cos_i + \text{rotate\_half}(q^{\text{rot}}_i) \odot \sin_i
$$

<span style="font-size: 14px;">where $\text{rotate\_half}([a, b]) = [-b, a]$ on the two halves of the $r$-channel rotated block, and $\cos_i, \sin_i$ are the doubled-length vectors of length $r$ built from the supplied $(n, r/2)$ tables. The untouched channels pass through. Finally, after GQA expansion and concatenation:</span>

$$
\text{attn}(x) = \text{softmax}\!\left(\frac{Q K^T}{\sqrt{D}} + M\right) V, \qquad y = \text{attn}(x) \cdot W_o + b_o
$$

---

## <span style="font-size: 16px;">Grouped-Query Attention</span>

<span style="font-size: 14px;">Standard Multi-Head Attention stores one K head and one V head per query head. For a 96-head 128-dim attention block over a 32K context, that is $96 \times 128 \times 32768 \times 2 \times 2$ bytes (fp16) per layer just for the KV cache: roughly 1.6 GB per layer, multiplied by ~64 layers in GLM-4.5. This dominates inference memory.</span>

* <span style="font-size: 14px;">**MQA** (Multi-Query Attention, Shazeer 2019) goes to one shared KV head, cutting cache by $H \times$ but losing quality.</span>
* <span style="font-size: 14px;">**GQA** (Ainslie et al., 2023) interpolates: $H_k$ KV heads, with each KV head shared by $g = H / H_k$ consecutive query heads. The cache shrinks by $g$ with almost no quality loss versus full MHA, as shown by the paper's PaLM finetuning experiments.</span>
* <span style="font-size: 14px;">GLM-4.5 uses $H = 96$, $H_k = 8$, $g = 12$. This is the same general regime as LLaMA-3 70B ($H=64, H_k=8, g=8$).</span>

<span style="font-size: 14px;">Implementation note: the expansion happens **after** RoPE and **before** the matmul. We typically use $\texttt{repeat\_interleave}$ so that head $q_i$ gets paired with KV head $\lfloor i / g \rfloor$. Using $\texttt{repeat}$ (block-tile) instead of $\texttt{repeat\_interleave}$ (element-tile) silently scrambles which query head reads from which KV head.</span>

---

## <span style="font-size: 16px;">QK-Norm: Why and Where</span>

<span style="font-size: 14px;">As models scale, $QK^T$ scores can grow large enough that softmax saturates to a one-hot distribution. Once that happens, gradients through the attention pattern vanish and training stalls or diverges. **QK-Norm** (Henry et al., 2020 first; Dehghani et al., 2023 popularized for ViT-22B) attacks this directly by normalizing $Q$ and $K$ to a bounded RMS before the dot product.</span>

* <span style="font-size: 14px;">$Q$ and $K$ are passed through RMSNorm with **their own learned per-channel gains** $\gamma_q, \gamma_k$. The gains are shared across heads in the GLM-4.5 implementation.</span>
* <span style="font-size: 14px;">Norm is applied along the head_dim axis, not the head axis. Each query vector becomes unit-RMS up to its gain.</span>
* <span style="font-size: 14px;">The scaling $1/\sqrt{D}$ is **kept** despite the normalization. Together they keep raw scores in a narrow numerical band even for very long contexts.</span>

<span style="font-size: 14px;">Crucially, **QK-Norm comes BEFORE RoPE** in GLM-4.5, matching the reference HuggingFace implementation: the `if self.use_qk_norm` block runs immediately after the QKV projection and before `apply_rotary_pos_emb`. Reversing this order changes the semantics: RoPE preserves norms per-token but mixes channels, so normalizing after RoPE would normalize a rotated mixture and not the raw projection.</span>

---

## <span style="font-size: 16px;">Partial RoPE (Rotate-Half Convention)</span>

<span style="font-size: 14px;">Rotary Position Embedding (Su et al., 2021) injects position by rotating pairs of channels in $Q$ and $K$ as a function of the token index. In the rotate-half implementation (used by LLaMA, GPT-NeoX, GLM-4.5), the rotated dimension $r$ is split in two halves and the rotation acts as:</span>

$$
q^{\text{rot}} = q \odot \cos + \text{rotate\_half}(q) \odot \sin
$$

<span style="font-size: 14px;">where $\text{rotate\_half}([h_1, h_2]) = [-h_2, h_1]$ joins the two halves along the channel axis. The supplied $(\cos, \sin)$ tables have width $r/2$ (one entry per frequency); they are doubled to width $r$ by concatenation $[\cos, \cos]$ before broadcasting over the head axis.</span>

<span style="font-size: 14px;">**Partial RoPE** restricts the rotation to the first $r$ channels of each $D$-dimensional head and leaves the last $D - r$ channels untouched. The intuition:</span>

* <span style="font-size: 14px;">RoPE gives **relative** position information through phase. Some channels benefit from this; others are better used as raw content channels uncoupled from position.</span>
* <span style="font-size: 14px;">Empirically, $r = D/2$ works well, and it halves the cost of the rotation. GLM-4.5 sets `partial_rotary_factor = 0.5`, so $r = 64$ for $D = 128$.</span>
* <span style="font-size: 14px;">This is the same convention used by GPT-NeoX, Falcon, and the Phi family.</span>

<span style="font-size: 14px;">An edge case worth handling explicitly: if $r = 0$, RoPE is skipped entirely and $Q, K$ pass through unchanged. If $r = D$, the rotation covers the full head dimension (standard full RoPE).</span>

---

## <span style="font-size: 16px;">Putting It Together: Order of Operations</span>

<span style="font-size: 14px;">The exact order matters and is easy to get wrong. From the HuggingFace `Glm4MoeAttention.forward` source (which the GLM-4.5 release uses):</span>

* <span style="font-size: 14px;">**1. Project**: $\texttt{q\_proj}, \texttt{k\_proj}, \texttt{v\_proj}$ from the same $x$. In this problem they are fused into one $W_{qkv}$.</span>
* <span style="font-size: 14px;">**2. View into heads**: reshape last axis to $(\text{heads}, D)$.</span>
* <span style="font-size: 14px;">**3. QK-Norm**: $\texttt{q = q\_norm(q)}; \texttt{k = k\_norm(k)}$ guarded by $\texttt{use\_qk\_norm}$.</span>
* <span style="font-size: 14px;">**4. RoPE**: $\texttt{apply\_rotary\_pos\_emb(q, k, cos, sin)}$ with rotate-half on the first `cos.shape[-1]` channels.</span>
* <span style="font-size: 14px;">**5. Causal SDPA + GQA**: $\texttt{repeat\_kv}$ inside the eager attention kernel expands K and V along the head axis by $g$, then $\text{softmax}(QK^T \cdot \text{scaling})$ with the causal mask.</span>
* <span style="font-size: 14px;">**6. Output projection**: $\texttt{o\_proj(attn\_output)}$. In GLM-4.5 the o_proj has no bias; this problem allows a bias slot.</span>

---

## <span style="font-size: 16px;">Numerical Example (n=2, H=2, Hk=1, D=4, r=4)</span>

<span style="font-size: 14px;">Take a 2-token sequence with two query heads, one KV head (group size 2), and a head dim of 4 with full RoPE. After the QKV projection we have</span>

$$
Q \in \mathbb{R}^{2 \times 2 \times 4}, \quad K, V \in \mathbb{R}^{2 \times 1 \times 4}
$$

<span style="font-size: 14px;">QK-Norm rescales each $(t, h)$ slot of $Q$ and $K$ to RMS = 1 (times the per-channel gain). RoPE then rotates the 4 channels of each head with the supplied length-2 $\cos, \sin$ vectors doubled to length 4. After GQA expansion $K, V$ become shape $(2, 2, 4)$. The $QK^T$ matrix per head is $2 \times 2$ and the causal mask zeroes the upper-triangular entry $[0, 1]$, leaving:</span>

* <span style="font-size: 14px;">Row 0 of the softmax has all mass on column 0 (only legal key).</span>
* <span style="font-size: 14px;">Row 1 mixes columns 0 and 1 by the softmax of $QK^T / \sqrt{4}$.</span>

<span style="font-size: 14px;">The two query heads share the same key cache (group size 2) but their **queries** are different, so the attention weights still differ across heads. The final $\text{attn} \in \mathbb{R}^{2 \times 8}$ goes through $W_o$ to produce the 8-dim output. The public test case `two_tokens_gqa2_full_rope` realizes this exact computation.</span>

---

## <span style="font-size: 16px;">Why These Choices Matter for GLM-4.5</span>

* <span style="font-size: 14px;">**GQA at $g=12$** keeps the KV cache for a single 128K-context conversation in a single A100's HBM, which would be infeasible under MHA.</span>
* <span style="font-size: 14px;">**QK-Norm** is one of the simplest tricks that keeps post-training reward modeling and DPO stable: large positive rewards no longer push raw scores to saturation regions.</span>
* <span style="font-size: 14px;">**Partial RoPE at 50%** is a deliberate departure from LLaMA's full RoPE. The GLM team and earlier GPT-NeoX-style models found that mixing rotated and unrotated channels eases extrapolation to longer contexts than seen during pretraining.</span>
* <span style="font-size: 14px;">**`attention_bias=True`** means both q_proj and k_proj keep their bias terms (unlike LLaMA which drops them). The bias is a small constant offset per channel; it shows up in this problem's $b_{qkv}$.</span>

---

## <span style="font-size: 16px;">Complexity</span>

* <span style="font-size: 14px;">QKV projection: $O(n \, d_{\text{model}} \, (H + 2 H_k) D)$.</span>
* <span style="font-size: 14px;">RMSNorm on Q, K: $O(n (H + H_k) D)$, dominated by the projection.</span>
* <span style="font-size: 14px;">RoPE on first $r$ channels: $O(n (H + H_k) r)$.</span>
* <span style="font-size: 14px;">SDPA (after GQA expansion): $O(n^2 H D)$ FLOPs and $O(n^2 H)$ memory for the score matrix in the naive form.</span>
* <span style="font-size: 14px;">Output projection: $O(n H D \, d_{\text{model}})$.</span>

<span style="font-size: 14px;">The dominant term at long context is the $O(n^2 H D)$ score matmul. GLM-4.5 uses FlashAttention-2 for this in practice; the public test cases are tiny enough that an eager implementation matches numerically.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Applying RoPE before QK-Norm.** The HuggingFace reference is explicit: QK-Norm first, then RoPE. Reversing the order rotates a normalized vector, which is a different mathematical operation. Test cases will fail by a small but persistent amount.</span>
* <span style="font-size: 14px;">**Rotating all channels instead of the first $r$.** Forgetting to slice $q[..., :r]$ produces wrong logits when $r < D$, silently. The shape stays correct, so this often passes a single test and fails on the partial-RoPE cases.</span>
* <span style="font-size: 14px;">**Using $\texttt{repeat}$ instead of $\texttt{repeat\_interleave}$ for GQA.** $\texttt{repeat(g, 1, 1)}$ block-tiles the KV heads ($\texttt{kv0, kv1, ..., kvHk-1, kv0, kv1, ...}$) while $\texttt{repeat\_interleave(g, dim=0)}$ element-tiles them ($\texttt{kv0, kv0, ..., kv1, kv1, ...}$). The latter is what pairs query head $q_i$ with KV head $\lfloor i / g \rfloor$.</span>
* <span style="font-size: 14px;">**Forgetting the causal mask.** Without $-\infty$ above the diagonal, the first token attends to future tokens. The model becomes acausal and training collapses. This passes shape checks and only fails values.</span>
* <span style="font-size: 14px;">**Dropping the $1/\sqrt{D}$ scale.** QK-Norm bounds the per-element magnitude but not the dot-product magnitude, which grows with $D$. Removing the scale makes softmax saturate on long sequences.</span>
* <span style="font-size: 14px;">**Forgetting the output projection $W_o$.** A common slip when refactoring: returning the concatenated head output directly. The shape happens to be $(n, H D)$, which equals $(n, d_{\text{model}})$ in many configs, so the bug compiles silently.</span>
* <span style="font-size: 14px;">**Wrong cos/sin shape handling.** The supplied tables have shape $(n, r/2)$; if you broadcast them straight onto $(n, H, r)$ without first concatenating along the last axis to length $r$, you get a shape error or a wrong rotation. The standard fix is $\texttt{cos\_full = torch.cat([cos, cos], dim=-1)}$ then unsqueeze the head axis.</span>
* <span style="font-size: 14px;">**Mixing up the rotate-half halves.** $\texttt{rotate\_half([a, b]) = [-b, a]}$. Returning $[b, -a]$ produces the conjugate rotation and breaks relative position math.</span>

---

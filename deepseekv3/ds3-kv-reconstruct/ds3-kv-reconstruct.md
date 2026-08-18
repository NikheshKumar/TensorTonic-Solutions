# <span style="font-size: 20px;">KV Reconstruction via Up-Projection</span>

<span style="font-size: 14px;">In DeepSeek V3's Multi-head Latent Attention (MLA), the KV cache stores only a compact latent vector $c_{kv}$ per token instead of full key and value tensors. KV reconstruction recovers usable key and value representations from this compressed latent by applying learned up-projection matrices. The key reconstruction produces only the non-positional component $K_{\text{nope}}$ with $d_{\text{nope}}$ dimensions per head, because positional information is handled separately through a decoupled RoPE pathway.</span>

<span style="font-size: 14px;">This is the second stage of the MLA pipeline. KV compression (stage one) projects the input down to $c_{kv}$, and KV reconstruction projects it back up. The dimensional asymmetry between K and V reconstruction is one of MLA's most distinctive design choices.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">KV reconstruction takes the compressed latent $c_{kv} \in \mathbb{R}^{B \times S \times d_c}$ and applies two separate learned up-projections to recover the key and value tensors for multi-head attention:</span>

* <span style="font-size: 14px;">**Key up-projection ($W_{uk}$):** Maps $c_{kv}$ to the non-positional key component $K_{\text{nope}}$. The output dimension is $n_h \cdot d_{\text{nope}}$, not $n_h \cdot d_h$, because each head's key only needs $d_{\text{nope}}$ dimensions from this pathway. The remaining $d_{\text{rope}}$ dimensions come from a completely separate decoupled RoPE mechanism.</span>
* <span style="font-size: 14px;">**Value up-projection ($W_{uv}$):** Maps $c_{kv}$ to the full value tensor $V$. The output dimension is $n_h \cdot d_h$, because values carry no positional encoding and need their complete per-head dimension.</span>

<span style="font-size: 14px;">After the linear projection, both outputs are reshaped from a flat vector into multi-head format and then transposed so the head dimension comes before the sequence dimension, producing the standard $(B, n_h, S, d)$ layout that attention expects.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">**K_nope reconstruction -- linear projection and reshape:**</span>

$$
K_{\text{nope}} = (c_{kv} \cdot W_{uk}^T).\text{view}(B, S, n_h, d_{\text{nope}}).\text{transpose}(1, 2)
$$

<span style="font-size: 14px;">Breaking this down step by step:</span>

* <span style="font-size: 14px;">$c_{kv} \in \mathbb{R}^{B \times S \times d_c}$ -- the compressed latent from KV compression.</span>
* <span style="font-size: 14px;">$W_{uk} \in \mathbb{R}^{(n_h \cdot d_{\text{nope}}) \times d_c}$ -- the key up-projection weight matrix.</span>
* <span style="font-size: 14px;">$c_{kv} \cdot W_{uk}^T$ produces shape $(B, S, n_h \cdot d_{\text{nope}})$ -- a flat vector per token containing all heads' key components packed together.</span>
* <span style="font-size: 14px;">$.view(B, S, n_h, d_{\text{nope}})$ splits the last dimension into separate heads, each with $d_{\text{nope}}$ dimensions.</span>
* <span style="font-size: 14px;">$.transpose(1, 2)$ swaps the sequence and head axes, yielding shape $(B, n_h, S, d_{\text{nope}})$.</span>

<span style="font-size: 14px;">**V reconstruction -- linear projection and reshape:**</span>

$$
V = (c_{kv} \cdot W_{uv}^T).\text{view}(B, S, n_h, d_h).\text{transpose}(1, 2)
$$

<span style="font-size: 14px;">The structure is identical but the dimensions differ:</span>

* <span style="font-size: 14px;">$W_{uv} \in \mathbb{R}^{(n_h \cdot d_h) \times d_c}$ -- the value up-projection weight matrix.</span>
* <span style="font-size: 14px;">$c_{kv} \cdot W_{uv}^T$ produces shape $(B, S, n_h \cdot d_h)$.</span>
* <span style="font-size: 14px;">$.view(B, S, n_h, d_h)$ splits into heads, each with $d_h$ dimensions (full head dimension).</span>
* <span style="font-size: 14px;">$.transpose(1, 2)$ produces shape $(B, n_h, S, d_h)$.</span>

<span style="font-size: 14px;">**Dimension summary:**</span>

* <span style="font-size: 14px;">$d_c$ -- compressed latent dimension (e.g., 512 in DeepSeek V3)</span>
* <span style="font-size: 14px;">$d_h$ -- full head dimension (e.g., 128)</span>
* <span style="font-size: 14px;">$d_{\text{nope}} = d_h - d_{\text{rope}}$ -- head dimension minus the RoPE dimensions (e.g., $128 - 64 = 64$)</span>
* <span style="font-size: 14px;">$n_h$ -- number of attention heads</span>
* <span style="font-size: 14px;">$W_{uk}$ output dim: $n_h \cdot d_{\text{nope}}$ (smaller than $W_{uv}$'s output because K only needs $d_{\text{nope}}$ per head)</span>
* <span style="font-size: 14px;">$W_{uv}$ output dim: $n_h \cdot d_h$ (full head dimension for values)</span>

---

## <span style="font-size: 16px;">Why Separate K and V Projections</span>

<span style="font-size: 14px;">Both K and V come from the same compressed representation $c_{kv}$, so why not use a single projection? The answer lies in the different roles keys and values play, and the dimensional constraints MLA imposes.</span>

<span style="font-size: 14px;">**Keys determine relevance.** Query-key dot products compute how much each position should attend to every other position. In MLA, the key is split into $K_{\text{nope}}$ (content-based similarity) and $K_{\text{rope}}$ (position-based similarity). Since $K_{\text{rope}}$ comes from a separate pathway, $W_{uk}$ only needs to produce $K_{\text{nope}}$, learning to extract exactly the content features that queries will match against.</span>

<span style="font-size: 14px;">**Values carry information.** After attention weights are computed, values are weighted-summed to form the output. They carry the actual information content flowing forward through the network, with no positional component and the full $d_h$ dimensions per head.</span>

<span style="font-size: 14px;">**Different output dimensions mandate separate weights.** Because $d_{\text{nope}} \neq d_h$, a single weight matrix cannot produce both K and V correctly. Even concatenating into one larger projection and splitting would be mathematically equivalent to two separate projections with no parameter sharing.</span>

<span style="font-size: 14px;">**Learned specialization.** $W_{uk}$ learns to extract similarity-matching features from $c_{kv}$, while $W_{uv}$ extracts information-carrying features. Each specializes while sharing the same compressed bottleneck.</span>

---

## <span style="font-size: 16px;">The d_nope Distinction</span>

<span style="font-size: 14px;">The key up-projection produces $d_{\text{nope}}$ dimensions per head, not $d_h$. This is a direct consequence of MLA's decoupled RoPE architecture.</span>

<span style="font-size: 14px;">**The full key is a concatenation.** In standard multi-head attention, each head's key has $d_h$ dimensions and RoPE is applied to all of them. In MLA, the full key for each head is:</span>

$$
K_{\text{full}} = \text{concat}(K_{\text{nope}},\; K_{\text{rope}}) \in \mathbb{R}^{B \times n_h \times S \times d_h}
$$

<span style="font-size: 14px;">where $K_{\text{nope}} \in \mathbb{R}^{B \times n_h \times S \times d_{\text{nope}}}$ and $K_{\text{rope}} \in \mathbb{R}^{B \times n_h \times S \times d_{\text{rope}}}$.</span>

<span style="font-size: 14px;">**Why the split exists.** The compressed latent $c_{kv}$ is cached during inference. If RoPE were applied before compression, the cached representation would contain position-dependent information. Different positions would have different latent representations even for identical tokens, destroying the compressibility that MLA relies on. MLA solves this by ensuring $c_{kv}$ is completely position-free. The key's positional component $K_{\text{rope}}$ is computed from a separate pathway (the decoupled RoPE mechanism) that takes the original input $x$ through a dedicated down-projection.</span>

<span style="font-size: 14px;">**Dimensional accounting.** Because $d_h = d_{\text{nope}} + d_{\text{rope}}$, the key up-projection only needs to produce $d_{\text{nope}}$ dimensions per head. The remaining $d_{\text{rope}}$ dimensions are filled by the decoupled RoPE pathway.</span>

<span style="font-size: 14px;">**Concrete example of the split.** With $d_h = 128$ and $d_{\text{rope}} = 64$:</span>

* <span style="font-size: 14px;">$d_{\text{nope}} = 128 - 64 = 64$ dimensions per head from $W_{uk}$ (content features, no position information)</span>
* <span style="font-size: 14px;">$d_{\text{rope}} = 64$ dimensions per head from the decoupled RoPE pathway (position information only)</span>
* <span style="font-size: 14px;">Full key: $64 + 64 = 128 = d_h$ dimensions per head after concatenation</span>

<span style="font-size: 14px;">**Value has no such split.** Values are weighted-summed using attention weights that already encode positional information via the Q-K dot product. The value up-projection produces the full $d_h$ dimensions per head with no positional split needed.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">KV reconstruction is part of the MLA mechanism introduced in DeepSeek V2 and refined in DeepSeek V3. MLA's core innovation is reducing the KV cache memory from $O(n_h \cdot d_h)$ per token to $O(d_c)$ per token, where $d_c \ll n_h \cdot d_h$. The full MLA pipeline has four stages:</span>

* <span style="font-size: 14px;">**Stage 1 -- KV Compression:** Project the input $x$ down to $c_{kv} = x \cdot W_{dkv}^T$ where $c_{kv} \in \mathbb{R}^{d_c}$. This is what gets cached during inference.</span>
* <span style="font-size: 14px;">**Stage 2 -- KV Reconstruction (this problem):** Apply $W_{uk}$ and $W_{uv}$ to recover $K_{\text{nope}}$ and $V$ from the cached $c_{kv}$.</span>
* <span style="font-size: 14px;">**Stage 3 -- Decoupled RoPE:** Compute $K_{\text{rope}}$ from a separate pathway and concatenate with $K_{\text{nope}}$ to form the full key. Apply RoPE to query components as well.</span>
* <span style="font-size: 14px;">**Stage 4 -- Full MLA Attention:** Compute attention using the reconstructed Q, K, V and produce the output.</span>

<span style="font-size: 14px;">The connection to compression is tight. During training, $W_{dkv}$ (compression), $W_{uk}$ (key reconstruction), and $W_{uv}$ (value reconstruction) are all learned jointly. This is analogous to an autoencoder: $W_{dkv}$ is the encoder, and $W_{uk}$, $W_{uv}$ together form the decoder. The bottleneck dimension $d_c$ controls the compression ratio.</span>

<span style="font-size: 14px;">During inference, only $c_{kv}$ (and the small $K_{\text{rope}}$ vector) need to be stored per token in the KV cache. The up-projection matrices $W_{uk}$ and $W_{uv}$ are model weights shared across all tokens, adding no per-token memory cost. The model reconstructs K and V on-the-fly from the cached latents, trading compute for memory -- a favorable tradeoff because modern GPUs are memory-bandwidth-bound during inference.</span>

<span style="font-size: 14px;">In DeepSeek V3, $d_c = 512$ while $n_h \cdot d_h = 128 \cdot 128 = 16384$. The KV cache stores 512 values per token instead of 32768 (K and V combined), a 64x compression ratio.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider a minimal configuration with $B = 1$, $S = 2$, $n_h = 2$ heads, $d_c = 4$, $d_h = 4$, $d_{\text{rope}} = 2$, so $d_{\text{nope}} = 4 - 2 = 2$.</span>

<span style="font-size: 14px;">**Input -- compressed latent $c_{kv}$:**</span>

$$
c_{kv} = \begin{bmatrix} 1.0 & 0.5 & -0.3 & 0.8 \\ 0.2 & -0.4 & 0.6 & 0.1 \end{bmatrix} \in \mathbb{R}^{2 \times 4}
$$

<span style="font-size: 14px;">Each row is one token's compressed representation ($S = 2$ tokens, $d_c = 4$).</span>

<span style="font-size: 14px;">**Key up-projection $W_{uk} \in \mathbb{R}^{4 \times 4}$** (output dim $= n_h \cdot d_{\text{nope}} = 2 \times 2 = 4$):</span>

$$
W_{uk} = \begin{bmatrix} 0.1 & 0.2 & 0.3 & 0.4 \\ -0.1 & 0.5 & 0.0 & 0.2 \\ 0.3 & -0.2 & 0.1 & 0.6 \\ 0.4 & 0.1 & -0.3 & 0.0 \end{bmatrix}
$$

<span style="font-size: 14px;">**Step 1 -- Linear projection:** $c_{kv} \cdot W_{uk}^T \in \mathbb{R}^{2 \times 4}$</span>

<span style="font-size: 14px;">Token 0: $[1.0, 0.5, -0.3, 0.8] \cdot W_{uk}^T$</span>

* <span style="font-size: 14px;">dim 0: $1.0(0.1) + 0.5(0.2) + (-0.3)(0.3) + 0.8(0.4) = 0.43$</span>
* <span style="font-size: 14px;">dim 1: $1.0(-0.1) + 0.5(0.5) + (-0.3)(0.0) + 0.8(0.2) = 0.31$</span>
* <span style="font-size: 14px;">dim 2: $1.0(0.3) + 0.5(-0.2) + (-0.3)(0.1) + 0.8(0.6) = 0.65$</span>
* <span style="font-size: 14px;">dim 3: $1.0(0.4) + 0.5(0.1) + (-0.3)(-0.3) + 0.8(0.0) = 0.54$</span>

<span style="font-size: 14px;">Token 1: $[0.2, -0.4, 0.6, 0.1] \cdot W_{uk}^T$</span>

* <span style="font-size: 14px;">dim 0: $0.2(0.1) + (-0.4)(0.2) + 0.6(0.3) + 0.1(0.4) = 0.16$</span>
* <span style="font-size: 14px;">dim 1: $0.2(-0.1) + (-0.4)(0.5) + 0.6(0.0) + 0.1(0.2) = -0.20$</span>
* <span style="font-size: 14px;">dim 2: $0.2(0.3) + (-0.4)(-0.2) + 0.6(0.1) + 0.1(0.6) = 0.26$</span>
* <span style="font-size: 14px;">dim 3: $0.2(0.4) + (-0.4)(0.1) + 0.6(-0.3) + 0.1(0.0) = -0.14$</span>

<span style="font-size: 14px;">Projected result shape $(1, 2, 4)$:</span>

$$
\begin{bmatrix} 0.43 & 0.31 & 0.65 & 0.54 \\ 0.16 & -0.20 & 0.26 & -0.14 \end{bmatrix}
$$

<span style="font-size: 14px;">**Step 2 -- Reshape:** $.view(1, 2, 2, 2)$ splits each 4-element vector into 2 heads with $d_{\text{nope}} = 2$ dims each:</span>

* <span style="font-size: 14px;">Token 0, Head 0: $[0.43, 0.31]$ | Head 1: $[0.65, 0.54]$</span>
* <span style="font-size: 14px;">Token 1, Head 0: $[0.16, -0.20]$ | Head 1: $[0.26, -0.14]$</span>

<span style="font-size: 14px;">**Step 3 -- Transpose:** $.transpose(1, 2)$ reorders to $(B, n_h, S, d_{\text{nope}})$:</span>

$$
K_{\text{nope}} = \begin{bmatrix} \text{Head 0:} & [0.43, 0.31], & [0.16, -0.20] \\ \text{Head 1:} & [0.65, 0.54], & [0.26, -0.14] \end{bmatrix}
$$

<span style="font-size: 14px;">Final shape: $(1, 2, 2, 2)$ -- batch 1, 2 heads, 2 tokens, $d_{\text{nope}} = 2$.</span>

<span style="font-size: 14px;">**Value reconstruction follows the same steps** but with $W_{uv} \in \mathbb{R}^{8 \times 4}$ ($n_h \cdot d_h = 2 \times 4 = 8$ output dims), reshaping to $.view(1, 2, 2, 4)$ then transposing to $(1, 2, 2, 4)$. Each head gets the full $d_h = 4$ dimensions.</span>

<span style="font-size: 14px;">**The critical observation:** $K_{\text{nope}}$ has last dimension $d_{\text{nope}} = 2$ while $V$ has last dimension $d_h = 4$. The key is incomplete -- it needs $d_{\text{rope}} = 2$ more dimensions from the decoupled RoPE pathway to form the full $d_h = 4$ dimensional key for attention.</span>

---

## <span style="font-size: 16px;">Common Pitfalls</span>

<span style="font-size: 14px;">**Pitfall 1 -- Wrong reshape order.** The sequence must be view first, then transpose. If you transpose $(B, S, n_h \cdot d) \to (B, n_h \cdot d, S)$ before reshaping, the view operation splits values that are now interleaved across the wrong axis. Always reshape while the data is still in $(B, S, \ldots)$ layout, then transpose $S$ and $n_h$.</span>

<span style="font-size: 14px;">**Pitfall 2 -- Using $d_h$ instead of $d_{\text{nope}}$ for the key reshape.** Reshaping the key output with $.view(B, S, n_h, d_h)$ instead of $.view(B, S, n_h, d_{\text{nope}})$ will fail because $W_{uk}$ outputs $n_h \cdot d_{\text{nope}}$ values, not $n_h \cdot d_h$. The element count does not match.</span>

<span style="font-size: 14px;">**Pitfall 3 -- Transposing the wrong dimensions.** The transpose must swap axes 1 and 2: $(B, S, n_h, d) \to (B, n_h, S, d)$. Transposing axes 2 and 3 would swap head and feature dimensions; transposing 0 and 1 would swap batch and sequence. The correct call is $.transpose(1, 2)$.</span>

<span style="font-size: 14px;">**Pitfall 4 -- Forgetting the RoPE component completes the key.** $K_{\text{nope}}$ is not ready for attention alone. It must be concatenated with $K_{\text{rope}}$ along the last dimension. Without concatenation, the query-key dot product uses only $d_{\text{nope}}$ dimensions, discarding all positional information.</span>

<span style="font-size: 14px;">**Pitfall 5 -- Confusing $W_{uk}$ shape convention.** In PyTorch's nn.Linear, weight shape is (out_features, in_features). So $W_{uk} \in \mathbb{R}^{(n_h \cdot d_{\text{nope}}) \times d_c}$ and the projection is $c_{kv} \cdot W_{uk}^T$. Getting the transpose wrong causes a shape mismatch or silently produces garbage.</span>

<span style="font-size: 14px;">**Pitfall 6 -- Assuming K and V have the same per-head dimension.** In standard MHA, K and V both use $d_h$ per head. In MLA, K uses $d_{\text{nope}}$ per head (with $d_{\text{rope}}$ appended later) while V uses the full $d_h$. Code that derives the V head dimension from the K projection's output will produce incorrect shapes.</span>
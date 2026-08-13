# <span style="font-size: 20px;">Full Gemma 3 Block</span>

<span style="font-size: 14px;">A Gemma 3 block is one complete repeating unit inside the model's Transformer stack. It chains two sub-layers: an attention sub-block followed by a SwiGLU feedforward network (FFN). Each sub-layer is wrapped in its own RMSNorm and has its own residual connection. The input to the block enters the first sub-layer, and the output of the second sub-layer becomes the input to the next block.</span>

<span style="font-size: 14px;">This is the "full picture" problem for Gemma 3. Earlier problems isolated individual components -- QK-Norm, sliding window attention, layer routing. Here, all of them are assembled into a single block. The attention sub-layer may use local (sliding window) or global attention depending on the layer index, but the FFN sub-layer is identical regardless.</span>

---

## <span style="font-size: 16px;">What It Is: One Complete Repeating Unit</span>

<span style="font-size: 14px;">The Gemma 3 model stacks $N$ identical-structure blocks. Each block transforms a hidden state $h$ of shape $(B, T, d)$ where $B$ is batch size, $T$ is sequence length, and $d$ is the model hidden dimension. The block has two sub-layers executed sequentially:</span>

* <span style="font-size: 14px;">**Sub-layer 1 -- Attention:** RMSNorm the input, run multi-head attention (with QK-Norm, GQA, and either sliding window or global masking), then add the result back to the input (first residual connection).</span>
* <span style="font-size: 14px;">**Sub-layer 2 -- FFN:** RMSNorm the output of sub-layer 1, run SwiGLU feedforward, then add the result back (second residual connection).</span>

<span style="font-size: 14px;">Each block has its own learned parameters: attention weights ($W_q, W_k, W_v, W_o$), QK-Norm scales ($\gamma_q, \gamma_k$), two RMSNorm weight vectors ($\gamma_{\text{attn}}, \gamma_{\text{ffn}}$), and three FFN matrices ($W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$). No parameters are shared between blocks.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">**Attention sub-layer (summarized):**</span>

$$
a = \text{Attention}(\text{RMSNorm}_{\text{attn}}(h))
$$

$$
h_{\text{mid}} = h + a
$$

<span style="font-size: 14px;">where $\text{Attention}(\cdot)$ includes Q/K/V projection, QK-Norm, optional RoPE, scaled dot-product attention with causal mask, and output projection. The attention internals were covered in the attention sub-block problem.</span>

<span style="font-size: 14px;">**RMSNorm before FFN:**</span>

$$
\hat{h} = \text{RMSNorm}_{\text{ffn}}(h_{\text{mid}}) = \frac{h_{\text{mid}} \cdot \gamma_{\text{ffn}}}{\sqrt{\frac{1}{d}\sum_{j=1}^{d} h_{\text{mid},j}^2 + \epsilon}}
$$

<span style="font-size: 14px;">where $\gamma_{\text{ffn}} \in \mathbb{R}^{d}$ is a learnable scale vector (separate from $\gamma_{\text{attn}}$) and $\epsilon \approx 10^{-6}$.</span>

<span style="font-size: 14px;">**SwiGLU FFN:**</span>

$$
\text{gate} = \text{swish}(\hat{h} \cdot W_{\text{gate}}^T)
$$

$$
\text{up} = \hat{h} \cdot W_{\text{up}}^T
$$

$$
\text{ffn}(\hat{h}) = (\text{gate} \odot \text{up}) \cdot W_{\text{down}}^T
$$

<span style="font-size: 14px;">where $\text{swish}(z) = z \cdot \sigma(z)$ and $\sigma$ is the sigmoid function.</span>

<span style="font-size: 14px;">**Second residual connection:**</span>

$$
h_{\text{out}} = h_{\text{mid}} + \text{ffn}(\hat{h})
$$

<span style="font-size: 14px;">The block output $h_{\text{out}}$ feeds into the next block (or into the final RMSNorm if this is the last block in the stack).</span>

---

## <span style="font-size: 16px;">The Two Sub-Layers</span>

<span style="font-size: 14px;">**Sub-layer 1: Attention.** The input $h$ is first normalized by $\text{RMSNorm}_{\text{attn}}$, then passed through multi-head attention. Gemma 3 uses Grouped Query Attention (GQA), QK-Norm on Q and K, and RoPE for positional encoding. The attention output is projected back to dimension $d$ via $W_o$ and added to the original $h$ through the first residual connection.</span>

<span style="font-size: 14px;">The critical structural point: the residual wraps around the entire attention sub-layer including the norm. The original unnormalized $h$ is what gets added, not the normalized version. This is the standard pre-norm Transformer pattern. In code: $h_{\text{mid}} = h + \text{Attn}(\text{RMSNorm}(h))$, not $h_{\text{mid}} = \text{RMSNorm}(h) + \text{Attn}(\text{RMSNorm}(h))$.</span>

<span style="font-size: 14px;">**Sub-layer 2: FFN.** The intermediate result $h_{\text{mid}}$ is normalized by a second, separate $\text{RMSNorm}_{\text{ffn}}$, then passed through SwiGLU. The FFN output is added to $h_{\text{mid}}$ (not to the original $h$) through the second residual connection, producing $h_{\text{out}}$.</span>

<span style="font-size: 14px;">Each sub-layer has its own independent RMSNorm with its own learnable $\gamma$ vector. These two norms are not shared. The attention norm stabilizes the input to attention; the FFN norm stabilizes the input to the feedforward network.</span>

---

## <span style="font-size: 16px;">SwiGLU FFN: Formula, Swish, and Gating</span>

<span style="font-size: 14px;">SwiGLU is a gated feedforward network that replaces the standard two-layer ReLU FFN used in the original Transformer. It uses three weight matrices instead of two, and the gating mechanism allows the network to selectively filter information.</span>

<span style="font-size: 14px;">**The three projections:**</span>

* <span style="font-size: 14px;">**Gate projection:** $W_{\text{gate}} \in \mathbb{R}^{d_{\text{ff}} \times d}$ projects the input to the intermediate dimension, then swish is applied. This produces a "gate" that controls which dimensions pass through.</span>
* <span style="font-size: 14px;">**Up projection:** $W_{\text{up}} \in \mathbb{R}^{d_{\text{ff}} \times d}$ projects the input to the same intermediate dimension with no activation. This is the "content" signal.</span>
* <span style="font-size: 14px;">**Down projection:** $W_{\text{down}} \in \mathbb{R}^{d \times d_{\text{ff}}}$ projects the gated result back to the model dimension $d$.</span>

<span style="font-size: 14px;">**The swish activation** is $\text{swish}(z) = z \cdot \sigma(z)$ where $\sigma(z) = 1/(1 + e^{-z})$. Swish is smooth and non-monotonic: it dips slightly below zero for small negative inputs before recovering. For large positive inputs, $\sigma(z) \approx 1$ so swish approaches the identity. This smooth behavior gives better gradient flow than ReLU, which has a hard zero for all negative values.</span>

<span style="font-size: 14px;">**The gating mechanism** multiplies gate and up outputs element-wise: $\text{gate} \odot \text{up}$. If the gate value for dimension $j$ is near zero, the content in that dimension is suppressed. If the gate value is large, the content passes through amplified. This is more expressive than a single activation because the gate and content come from different learned projections of the same input.</span>

<span style="font-size: 14px;">**Why SwiGLU outperforms ReLU FFN.** The standard FFN computes $\text{ReLU}(x W_1^T) W_2^T$. ReLU zeros out all negative pre-activations, so roughly half the neurons produce zero gradient. SwiGLU avoids this "dead neuron" problem because swish allows small negative values through. The gating mechanism provides data-dependent learned filtering that a fixed threshold cannot match. Empirically, SwiGLU achieves better perplexity per parameter, which is why Gemma 3, LLaMA, and most modern LLMs adopt it.</span>

<span style="font-size: 14px;">**Parameter cost.** A ReLU FFN has two matrices totaling $2 \cdot d \cdot d_{\text{ff}}$ parameters. SwiGLU has three, totaling $3 \cdot d \cdot d_{\text{ff}}$. To keep parameter count comparable, many models reduce $d_{\text{ff}}$ when using SwiGLU.</span>

---

## <span style="font-size: 16px;">How It Differs from a Standard Transformer Block</span>

<span style="font-size: 14px;">A Gemma 3 block shares the same high-level pre-norm structure, but differs in several internal details:</span>

* <span style="font-size: 14px;">**QK-Norm inside attention.** Standard Transformers go directly from Q/K projection to the dot product. Gemma 3 inserts RMSNorm on Q and K after projection but before RoPE, preventing attention logit explosion in deep networks.</span>
* <span style="font-size: 14px;">**Grouped Query Attention (GQA).** Standard MHA gives each head its own Q, K, V. GQA shares K/V heads across groups of query heads. Gemma 3 4B uses 8 query heads and 4 KV heads, so pairs of query heads share a single key-value pair.</span>
* <span style="font-size: 14px;">**Sliding window on local layers.** Standard Transformers use full causal attention at every layer. Gemma 3 alternates: most layers use local sliding window attention (window size 512 in 4B), every fifth layer uses global attention. This reduces KV cache memory while preserving long-range information flow.</span>
* <span style="font-size: 14px;">**SwiGLU instead of ReLU FFN.** Three-matrix gated FFN with swish activation replaces the two-matrix ReLU FFN.</span>
* <span style="font-size: 14px;">**RMSNorm instead of LayerNorm.** No mean subtraction and no bias term, which is computationally cheaper and works comparably well in practice.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">Gemma 3 (Google DeepMind, 2025) is a family of open-weight models from 1B to 27B parameters. Key dimensions:</span>

* <span style="font-size: 14px;">**Gemma 3 4B:** $d = 2560$, $d_{\text{ff}} = 10240$, 26 layers, 8 query heads, 4 KV heads, $d_{\text{head}} = 256$.</span>
* <span style="font-size: 14px;">**Gemma 3 12B:** $d = 3840$, $d_{\text{ff}} = 15360$, 48 layers, 16 query heads, 8 KV heads, $d_{\text{head}} = 256$.</span>
* <span style="font-size: 14px;">**Gemma 3 27B:** $d = 4608$, $d_{\text{ff}} = 36864$, 62 layers, 32 query heads, 16 KV heads, $d_{\text{head}} = 128$.</span>

<span style="font-size: 14px;">The FFN dimension is $4d$ for 4B and 12B, and $8d$ for 27B. The SwiGLU architecture is the same across all sizes. Every layer uses the same FFN regardless of whether attention is local or global. The local/global alternation pattern (every fifth layer is global) is consistent across model sizes.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Trace a single token through one full block with $d = 4$. Start with a hidden state from the previous block.</span>

<span style="font-size: 14px;">**Input:** $h = [1.0, \; -0.5, \; 2.0, \; 0.3]$</span>

### <span style="font-size: 14px;">Step 1: Attention sub-layer (summarized)</span>

<span style="font-size: 14px;">Assume the attention mechanism (RMSNorm, QK-Norm, GQA, softmax, output projection) produces:</span>

$$
a = [0.2, \; -0.1, \; 0.4, \; -0.05]
$$

<span style="font-size: 14px;">First residual connection:</span>

$$
h_{\text{mid}} = h + a = [1.0 + 0.2, \; -0.5 - 0.1, \; 2.0 + 0.4, \; 0.3 - 0.05] = [1.2, \; -0.6, \; 2.4, \; 0.25]
$$

### <span style="font-size: 14px;">Step 2: RMSNorm before FFN</span>

<span style="font-size: 14px;">Compute the RMS of $h_{\text{mid}}$:</span>

$$
\text{mean}(h_{\text{mid}}^2) = \frac{1.44 + 0.36 + 5.76 + 0.0625}{4} = \frac{7.6225}{4} = 1.9056
$$

$$
\text{RMS} = \sqrt{1.9056 + 10^{-6}} \approx 1.3805
$$

<span style="font-size: 14px;">With $\gamma_{\text{ffn}} = [1.0, \; 1.0, \; 1.0, \; 1.0]$ (at initialization):</span>

$$
\hat{h} = \frac{[1.2, \; -0.6, \; 2.4, \; 0.25]}{1.3805} \approx [0.8692, \; -0.4346, \; 1.7384, \; 0.1811]
$$

### <span style="font-size: 14px;">Step 3: SwiGLU FFN</span>

<span style="font-size: 14px;">Use $d_{\text{ff}} = 6$ for this example. Suppose $\hat{h} \cdot W_{\text{gate}}^T$ gives:</span>

$$
z_{\text{gate}} = [0.5, \; -1.2, \; 2.0, \; 0.1, \; -0.3, \; 1.5]
$$

<span style="font-size: 14px;">Apply swish element-wise ($\text{swish}(z) = z \cdot \sigma(z)$):</span>

* <span style="font-size: 14px;">$\text{swish}(0.5) = 0.5 \times 0.6225 = 0.3112$</span>
* <span style="font-size: 14px;">$\text{swish}(-1.2) = -1.2 \times 0.2315 = -0.2778$</span>
* <span style="font-size: 14px;">$\text{swish}(2.0) = 2.0 \times 0.8808 = 1.7616$</span>
* <span style="font-size: 14px;">$\text{swish}(0.1) = 0.1 \times 0.5250 = 0.0525$</span>
* <span style="font-size: 14px;">$\text{swish}(-0.3) = -0.3 \times 0.4256 = -0.1277$</span>
* <span style="font-size: 14px;">$\text{swish}(1.5) = 1.5 \times 0.8176 = 1.2263$</span>

$$
\text{gate} = [0.3112, \; -0.2778, \; 1.7616, \; 0.0525, \; -0.1277, \; 1.2263]
$$

<span style="font-size: 14px;">Suppose $\hat{h} \cdot W_{\text{up}}^T$ gives:</span>

$$
\text{up} = [0.8, \; 1.5, \; -0.3, \; 2.1, \; 0.4, \; -0.7]
$$

<span style="font-size: 14px;">Element-wise gating:</span>

$$
\text{gate} \odot \text{up} = [0.2490, \; -0.4167, \; -0.5285, \; 0.1103, \; -0.0511, \; -0.8584]
$$

<span style="font-size: 14px;">Down projection maps from $d_{\text{ff}} = 6$ back to $d = 4$. Suppose the result is:</span>

$$
\text{ffn}(\hat{h}) = [-0.15, \; 0.30, \; -0.08, \; 0.12]
$$

### <span style="font-size: 14px;">Step 4: Second residual connection</span>

$$
h_{\text{out}} = h_{\text{mid}} + \text{ffn}(\hat{h}) = [1.2 - 0.15, \; -0.6 + 0.30, \; 2.4 - 0.08, \; 0.25 + 0.12]
$$

$$
h_{\text{out}} = [1.05, \; -0.30, \; 2.32, \; 0.37]
$$

<span style="font-size: 14px;">This $h_{\text{out}}$ becomes the input to the next block. Both residuals preserve the general magnitude of the original input while making incremental refinements -- characteristic of well-trained Transformer blocks.</span>

---

## <span style="font-size: 16px;">Common Pitfalls</span>

<span style="font-size: 14px;">**Wrong residual connection targets.** The first residual adds attention output to $h$, producing $h_{\text{mid}}$. The second adds FFN output to $h_{\text{mid}}$, not to $h$. Adding both sub-layer outputs to the original $h$ bypasses the sequential composition. The two residuals are chained: $h \to h_{\text{mid}} \to h_{\text{out}}$.</span>

<span style="font-size: 14px;">**Missing the RMSNorm between sub-layers.** There must be an RMSNorm before attention AND another before the FFN. Skipping the second norm means the FFN receives unnormalized input whose magnitude depends on the attention output scale, causing training instability.</span>

<span style="font-size: 14px;">**Sharing RMSNorm parameters between sub-layers.** The two norms have independent learnable $\gamma$ vectors. Sharing $\gamma$ constrains the model to normalize attention input and FFN input identically, reducing representational capacity. Each $\gamma$ has $d$ parameters.</span>

<span style="font-size: 14px;">**Confusing gate and up projections.** The gate projection receives the swish activation; the up projection is linear (no activation). Swapping them changes the network's behavior: the gate should produce values in a controlled range (swish saturates for large positive inputs and approaches zero for large negative inputs), while the up projection provides unfiltered content. Reversing them means the "gate" has unbounded magnitude.</span>

<span style="font-size: 14px;">**Forgetting FFN is the same for local and global layers.** The attention sub-layer changes between local and global (different masks, different context windows). But the SwiGLU FFN is identical in structure and parameter shapes across all layers. Only the attention part varies with the local/global pattern.</span>

<span style="font-size: 14px;">**Normalizing after instead of before.** Gemma 3 uses pre-norm: RMSNorm is applied to the input of each sub-layer, not the output. Applying norm after attention instead of before changes the residual path's numerical properties and alters training dynamics significantly.</span>
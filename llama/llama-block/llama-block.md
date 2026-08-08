# <span style="font-size: 20px;">Llama 3 Transformer Block</span>

<span style="font-size: 14px;">The Llama 3 Transformer block is the core repeating unit in Meta's LLaMA family of large language models. Each block applies a 6-step pipeline: RMSNorm, Grouped-Query Attention with Rotary Position Embeddings, a residual addition, another RMSNorm, a SwiGLU feed-forward network, and a second residual addition. Stacking 32, 40, or 80 copies of this block produces the full LLaMA decoder.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">A single LLaMA block is one complete repeating unit in the decoder stack. The full model stacks $N$ identical copies sequentially, each with its own learned parameters but following the same structural template. Every token representation that enters the block exits as a refined representation that has been normalized, mixed with other positions via attention, reconnected through a residual, normalized again, transformed per-position through a gated FFN, and reconnected through a second residual.</span>

<span style="font-size: 14px;">The block takes a hidden state tensor of shape $(B, T, d)$ and produces an output of the same shape. Nothing about the sequence length or batch size changes -- the block is purely a refinement operation. The 6 steps form two sub-layers, each following the pattern: **normalize, transform, add residual**. The first sub-layer handles inter-token communication (attention), while the second handles per-token computation (FFN). This pre-norm residual design ensures stable gradient flow even when stacking 80+ blocks.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">**Step 1 -- Attention RMSNorm.** Normalize the input $h^{(\ell)}$ using RMSNorm:</span>

$$
\hat{h}^{(\ell)} = \frac{h^{(\ell)}}{\text{RMS}(h^{(\ell)})} \odot \gamma_1
$$

<span style="font-size: 14px;">where $\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}$ with $\epsilon = 10^{-5}$, and $\gamma_1 \in \mathbb{R}^d$ is a learned scale vector. There is no mean subtraction and no bias term.</span>

<span style="font-size: 14px;">**Step 2 -- Grouped-Query Attention with RoPE.** Project the normalized input into Q, K, V, apply RoPE to Q and K, then compute causal attention:</span>

$$
Q = \hat{h}^{(\ell)} W_Q, \quad K = \hat{h}^{(\ell)} W_K, \quad V = \hat{h}^{(\ell)} W_V
$$

$$
Q_{\text{rot}} = \text{RoPE}(Q), \quad K_{\text{rot}} = \text{RoPE}(K)
$$

$$
\text{GQA} = \text{softmax}\!\left(\frac{Q_{\text{rot}} K_{\text{rot}}^T}{\sqrt{d_h}} + M\right) V
$$

<span style="font-size: 14px;">where $W_Q \in \mathbb{R}^{d \times d}$, $W_K \in \mathbb{R}^{d \times d_{kv}}$, $W_V \in \mathbb{R}^{d \times d_{kv}}$, $d_{kv} = n_{kv} \times d_h$, $n_{kv}$ is the number of KV heads (fewer than query heads $n_h$), $M$ is a causal mask setting future positions to $-\infty$, and each KV head is shared across $n_h / n_{kv}$ query heads. The output is projected back through $W_O \in \mathbb{R}^{d \times d}$.</span>

<span style="font-size: 14px;">**Step 3 -- Attention residual:**</span>

$$
h^{(\ell+0.5)} = h^{(\ell)} + W_O \cdot \text{GQA}
$$

<span style="font-size: 14px;">The residual connects to the pre-norm input $h^{(\ell)}$, not the normalized $\hat{h}^{(\ell)}$. This is critical for maintaining the gradient highway.</span>

<span style="font-size: 14px;">**Step 4 -- FFN RMSNorm.** Normalize the post-attention state with a second, separate RMSNorm:</span>

$$
\hat{h}^{(\ell+0.5)} = \frac{h^{(\ell+0.5)}}{\text{RMS}(h^{(\ell+0.5)})} \odot \gamma_2
$$

<span style="font-size: 14px;">where $\gamma_2 \in \mathbb{R}^d$ is independent from $\gamma_1$. Each sub-layer has its own norm parameters.</span>

<span style="font-size: 14px;">**Step 5 -- SwiGLU FFN:**</span>

$$
\text{SwiGLU}(x) = (\text{SiLU}(x W_{\text{gate}}) \odot x W_{\text{up}}) W_{\text{down}}
$$

<span style="font-size: 14px;">where $\text{SiLU}(z) = z \cdot \sigma(z)$, $W_{\text{gate}}, W_{\text{up}} \in \mathbb{R}^{d \times d_{ff}}$, $W_{\text{down}} \in \mathbb{R}^{d_{ff} \times d}$, $d_{ff} = \frac{8}{3}d$ (rounded to the nearest multiple of 256), and no bias terms anywhere.</span>

<span style="font-size: 14px;">**Step 6 -- FFN residual:**</span>

$$
h^{(\ell+1)} = h^{(\ell+0.5)} + \text{SwiGLU}(\hat{h}^{(\ell+0.5)})
$$

---

## <span style="font-size: 16px;">Step-by-Step Walkthrough</span>

<span style="font-size: 14px;">**Step 1 -- RMSNorm (attention).** Before attention processes the input, RMSNorm stabilizes the activation scale. Unlike LayerNorm, it does not subtract the mean -- it only divides by the root-mean-square, then applies a learned element-wise scale. This ensures Q, K, V projections receive inputs at consistent magnitude regardless of what accumulated in the residual stream from earlier layers.</span>

<span style="font-size: 14px;">**Step 2 -- GQA with RoPE.** This is the inter-token communication step. After projecting Q, K, V, Rotary Position Embeddings are applied to Q and K (not V). RoPE rotates pairs of dimensions by position-dependent angles so that the Q-K dot product naturally encodes relative position. GQA then computes scaled dot-product attention with fewer KV heads than query heads, reducing KV cache size during inference while preserving quality. A causal mask restricts each token to attending only to itself and earlier positions.</span>

<span style="font-size: 14px;">**Step 3 -- Residual (attention).** The attention output (after $W_O$ projection) is added to the block's original input $h^{(\ell)}$. This skip connection provides a direct gradient path to earlier layers and lets each sub-layer learn a delta rather than a full representation.</span>

<span style="font-size: 14px;">**Step 4 -- RMSNorm (FFN).** A separate RMSNorm with its own learned $\gamma_2$ normalizes the post-attention state before the FFN. This is needed because the residual addition in Step 3 changes the hidden state scale, and the FFN projections need consistently scaled inputs.</span>

<span style="font-size: 14px;">**Step 5 -- SwiGLU FFN.** Each position is processed independently through a gated MLP. Two parallel projections (gate and up) produce intermediate vectors; the gate passes through SiLU activation, then is multiplied element-wise with the up projection. This gating lets the network learn which features to pass through and which to suppress. The result is projected back to model dimension by the down projection. The intermediate dimension is $\frac{8}{3}d$ instead of $4d$ to compensate for having three weight matrices instead of two.</span>

<span style="font-size: 14px;">**Step 6 -- Residual (FFN).** The FFN output is added to $h^{(\ell+0.5)}$, producing $h^{(\ell+1)}$ for the next block. The pattern is symmetric: both sub-layers normalize, transform, and add back to the residual stream.</span>

---

## <span style="font-size: 16px;">How It Differs from a GPT-2 Block</span>

<span style="font-size: 14px;">**RMSNorm vs. LayerNorm.** GPT-2 uses LayerNorm: $\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$. LLaMA uses RMSNorm: $\gamma \odot \frac{x}{\text{RMS}(x)}$ with no mean subtraction and no bias. RMSNorm is ~10-15% faster and empirically just as effective -- the mean subtraction in LayerNorm contributes little to training stability.</span>

<span style="font-size: 14px;">**GQA vs. MHA.** GPT-2 uses full Multi-Head Attention where every head has its own K and V projections. LLaMA uses Grouped-Query Attention with fewer KV heads than query heads. Llama 3 8B uses 32 query heads but only 8 KV heads, so each KV head serves 4 query heads. This reduces KV cache memory by $n_h / n_{kv}$ times during autoregressive inference with nearly identical quality to full MHA.</span>

<span style="font-size: 14px;">**SwiGLU vs. GELU FFN.** GPT-2: $\text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2$ with $d_{ff} = 4d$ and two weight matrices. LLaMA: $\text{SwiGLU}(x) = (\text{SiLU}(xW_{\text{gate}}) \odot xW_{\text{up}})W_{\text{down}}$ with $d_{ff} = \frac{8}{3}d$, three weight matrices, and no bias. The gating mechanism acts as a learned multiplicative filter, selectively passing or suppressing features.</span>

<span style="font-size: 14px;">**RoPE vs. learned positions.** GPT-2 adds a learned positional embedding once at the input before any block. Position information must propagate through all layers. LLaMA applies RoPE inside every attention layer, rotating Q and K so that dot products naturally encode relative distance $i - j$. RoPE has no learned parameters, is injected freshly at every layer, and can extrapolate to longer sequences than seen during training.</span>

<span style="font-size: 14px;">**No bias terms.** GPT-2 includes bias in all linear layers and both LayerNorm parameters ($\gamma$ and $\beta$). LLaMA removes all bias parameters everywhere: no bias in attention projections, no bias in FFN projections, and no $\beta$ in RMSNorm.</span>

---

## <span style="font-size: 16px;">The Two Sub-Layers</span>

<span style="font-size: 14px;">**Sub-layer 1: Attention (inter-token mixing).** Attention is the only component that allows different positions to exchange information. It computes weighted averages of value vectors, where weights are determined by query-key compatibility. This is a linear mixing operation -- it can gather relevant information from other positions but cannot apply nonlinear transformations to what it gathers. Its role is purely to move information between positions.</span>

<span style="font-size: 14px;">**Sub-layer 2: FFN (per-token computation).** The FFN processes each position independently, applying nonlinear transformations through SwiGLU gating. It transforms gathered information into useful features and acts as a key-value memory, mapping input patterns to output patterns through learned weights. SwiGLU provides two sources of nonlinearity: SiLU activation and element-wise gating multiplication.</span>

<span style="font-size: 14px;">**Pre-norm with residual.** Both sub-layers normalize before the transformation, not after. In post-norm (original Transformer), the residual passes through normalization, dampening gradients. In pre-norm, the residual stream flows through clean addition operations, creating an unobstructed gradient highway from the loss to the first layer. This is why LLaMA stacks up to 80 blocks without training instability.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">LLaMA was introduced in Touvron et al., "LLaMA: Open and Efficient Foundation Language Models" (2023). The paper demonstrated that smaller models trained on significantly more tokens can match or exceed larger models trained on fewer tokens. The architectural choices were drawn from prior work: RMSNorm from Zhang & Sennrich (2019), SwiGLU from Shazeer (2020), and RoPE from Su et al. (2021).</span>

<span style="font-size: 14px;">LLaMA stacks the block different numbers of times by model size:</span>

* <span style="font-size: 14px;">**LLaMA 7B:** $N = 32$ blocks, $d = 4096$, $n_h = 32$ heads, $d_h = 128$</span>
* <span style="font-size: 14px;">**LLaMA 13B:** $N = 40$ blocks, $d = 5120$, $n_h = 40$ heads, $d_h = 128$</span>
* <span style="font-size: 14px;">**LLaMA 65B:** $N = 80$ blocks, $d = 8192$, $n_h = 64$ heads, $d_h = 128$</span>

<span style="font-size: 14px;">The head dimension $d_h = 128$ is constant across all sizes. The FFN intermediate dimension is $d_{ff} = \frac{8}{3}d$ rounded to the nearest multiple of 256 (e.g., for $d = 4096$: $\frac{8}{3} \times 4096 = 10922.67 \to 11008$). Llama 2 added GQA for its 70B model, and Llama 3 uses GQA across all sizes. The pre-norm architecture means there is a final RMSNorm after the last block but before the LM head, separate from any block's internal norms.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Trace a sequence of 3 tokens through one LLaMA block with $d = 4$, $n_h = 2$ query heads, $n_{kv} = 1$ KV head, $d_h = 2$, $d_{ff} = 3$.</span>

<span style="font-size: 14px;">**Input** $h^{(\ell)}$: $h_0 = [1.0, -0.5, 0.8, 0.2]$, $h_1 = [-0.3, 0.7, 0.1, -0.6]$, $h_2 = [0.5, 0.3, -0.4, 0.9]$.</span>

<span style="font-size: 14px;">**Step 1 -- RMSNorm.** For $h_0$: $\text{RMS} = \sqrt{(1.0 + 0.25 + 0.64 + 0.04)/4} = \sqrt{0.4825} = 0.695$. With $\gamma_1 = [1, 1, 1, 1]$: $\hat{h}_0 = [1.439, -0.719, 1.151, 0.288]$. Similarly compute for $h_1$ and $h_2$.</span>

<span style="font-size: 14px;">**Step 2 -- GQA with RoPE.** After projection and RoPE rotation, suppose for head 1 at position 2: scores $= Q_2 K^T / \sqrt{2} = [0.198, 0.212, -0.028]$. After causal masking (all 3 positions visible to position 2) and softmax: weights $= [0.338, 0.343, 0.319]$. Weighted sum over V gives head output $[0.096, 0.141]$. Concatenating both heads and projecting through $W_O$ yields attention output $a_2 = [0.12, -0.08, 0.15, 0.03]$.</span>

<span style="font-size: 14px;">**Step 3 -- Residual.** $h^{(\ell+0.5)}_2 = h_2 + a_2 = [0.62, 0.22, -0.25, 0.93]$. Added to the original $h_2$, not the normalized version.</span>

<span style="font-size: 14px;">**Step 4 -- RMSNorm (FFN).** $\text{RMS} = \sqrt{(0.384 + 0.048 + 0.063 + 0.865)/4} = \sqrt{0.340} = 0.583$. $\hat{h}_2 = [1.063, 0.377, -0.429, 1.595]$.</span>

<span style="font-size: 14px;">**Step 5 -- SwiGLU.** Suppose gate $= [0.5, -0.3, 0.8]$ and up $= [0.2, 0.6, -0.1]$. SiLU: $\text{SiLU}(0.5) = 0.5 \times 0.622 = 0.311$, $\text{SiLU}(-0.3) = -0.3 \times 0.426 = -0.128$, $\text{SiLU}(0.8) = 0.8 \times 0.690 = 0.552$. Element-wise product with up: $[0.062, -0.077, -0.055]$. After $W_{\text{down}}$: $f_2 = [0.05, -0.03, 0.07, -0.02]$.</span>

<span style="font-size: 14px;">**Step 6 -- Residual.** $h^{(\ell+1)}_2 = [0.62 + 0.05,\; 0.22 - 0.03,\; -0.25 + 0.07,\; 0.93 - 0.02] = [0.67, 0.19, -0.18, 0.91]$. This exits the block into the next layer.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Wrong norm placement (post-norm instead of pre-norm).** LLaMA uses pre-norm: normalize before the sub-layer, then add the residual. Post-norm computes $h + \text{Norm}(\text{SubLayer}(h))$ which dampens the residual gradient path. Pre-norm computes $h + \text{SubLayer}(\text{Norm}(h))$, keeping the residual stream clean. Getting this wrong fundamentally changes training dynamics.</span>

* <span style="font-size: 14px;">**Wrong residual connection point.** The residual must add to the pre-norm input. Correct: $h^{(\ell+0.5)} = h^{(\ell)} + \text{Attn}(\text{RMSNorm}(h^{(\ell)}))$. Wrong: $h^{(\ell+0.5)} = \hat{h}^{(\ell)} + \text{Attn}(\hat{h}^{(\ell)})$. The second form breaks the gradient highway by routing the residual through normalization.</span>

* <span style="font-size: 14px;">**Sharing norm parameters between sub-layers.** Each block has two separate RMSNorm instances with independent $\gamma$ vectors. The attention norm ($\gamma_1$) and the FFN norm ($\gamma_2$) must not be shared -- they normalize representations at different points in the block.</span>

* <span style="font-size: 14px;">**Forgetting RoPE in attention.** RoPE must be applied to Q and K after projection but before computing attention scores. Without RoPE, the model has no position information at all (unlike GPT-2, where position is added at the input embedding). RoPE is applied to Q and K only, never to V.</span>

* <span style="font-size: 14px;">**Forgetting the causal mask.** Even with RoPE providing position awareness, the causal mask is still required for autoregressive modeling. Without it, each token attends to future tokens, leaking information during training. The mask sets scores for positions $j > i$ to $-\infty$ before softmax.</span>

* <span style="font-size: 14px;">**Wrong FFN expansion ratio.** LLaMA uses $d_{ff} = \frac{8}{3}d$ (not $4d$) because SwiGLU has three weight matrices instead of two. Using $4d$ with SwiGLU inflates the parameter count by ~50% compared to the intended design.</span>

* <span style="font-size: 14px;">**Adding bias terms.** LLaMA has no bias anywhere: no bias in $W_Q, W_K, W_V, W_O$, no bias in $W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$, and no $\beta$ in RMSNorm. Adding biases changes the function and breaks compatibility with pretrained weights.</span>

---
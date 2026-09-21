# <span style="font-size: 20px;">Partial RoPE</span>

<span style="font-size: 14px;">Partial Rotary Position Embedding (Partial RoPE) is the positional encoding used by GLM-4.5 (Zeng et al., 2025). It applies the standard rotate-half RoPE to only the first $\texttt{rope\_dim}$ channels of each attention head and leaves the remaining $\texttt{head\_dim} - \texttt{rope\_dim}$ channels untouched. The rotated channels carry position-dependent phase, while the pass-through channels carry position-agnostic features.</span>

---

## <span style="font-size: 16px;">What Partial RoPE Does</span>

<span style="font-size: 14px;">Standard RoPE (Su et al., 2021) rotates every channel of $Q$ and $K$ by a position-dependent angle. Partial RoPE keeps the same rotation formula but restricts it to the leading slice of each head, leaving the trailing slice as a clean linear subspace.</span>

* <span style="font-size: 14px;">**Rotated slice:** channels $0$ to $\texttt{rope\_dim} - 1$ get the rotate-half RoPE update using the precomputed $\cos$ and $\sin$ tables.</span>
* <span style="font-size: 14px;">**Pass-through slice:** channels $\texttt{rope\_dim}$ to $\texttt{head\_dim} - 1$ are copied to the output verbatim. No rotation, no scaling.</span>
* <span style="font-size: 14px;">**Same shape:** the output has identical shape to the input. Only the values in the leading slice change.</span>
* <span style="font-size: 14px;">**Edge case:** if $\texttt{rope\_dim} = \texttt{head\_dim}$, this reduces to full RoPE (the LLaMA convention). If $\texttt{rope\_dim} = 0$, it is a pure pass-through.</span>

---

## <span style="font-size: 16px;">Why Partial RoPE</span>

<span style="font-size: 14px;">Empirical analyses of RoPE-trained transformers (GLM team, GPT-NeoX, GLM-4) show that low-frequency rotary channels carry most of the useful relative-position signal. High-frequency channels rotate so fast that consecutive tokens are already near-orthogonal, which wastes head capacity that could instead carry semantic features. Partial RoPE makes this split explicit:</span>

* <span style="font-size: 14px;">**Concentrate position info in the first slice.** The $\cos$ and $\sin$ tables are derived from frequencies $1 / \theta^{2i / \texttt{rope\_dim}}$ for $i \in [0, \texttt{rope\_dim}/2)$. With a smaller rotary slice, the lowest frequency is even lower (longer wavelength), which improves long-context generalisation.</span>
* <span style="font-size: 14px;">**Free up the second slice for content.** The pass-through channels look like a small per-head linear residual that the model can use for token identity, value statistics, or learned biases without any positional interference.</span>
* <span style="font-size: 14px;">**Better extrapolation.** A narrower rotary band means fewer high-frequency oscillations to lose during inference at lengths beyond the training context. This matches the motivation behind YaRN and NTK-aware scaling.</span>
* <span style="font-size: 14px;">**Cheap.** It costs nothing extra at runtime: one fewer slice multiplication compared with full RoPE.</span>

<span style="font-size: 14px;">GLM-4.5 uses partial RoPE with $\texttt{rope\_dim}$ set to half the head dimension. The exact ratio is a configuration hyperparameter ($\texttt{partial\_rotary\_factor}$ in HuggingFace) that the canonical $\texttt{modeling\_glm4\_moe.py}$ exposes.</span>

---

## <span style="font-size: 16px;">The Math</span>

<span style="font-size: 14px;">Let $q \in \mathbb{R}^{N \times H \times D}$ where $N$ is the number of tokens, $H$ is the number of heads, and $D = \texttt{head\_dim}$. Split $q$ along the last axis:</span>

$$
q = [\, q_{\text{rope}} \;\|\; q_{\text{pass}} \,], \qquad q_{\text{rope}} \in \mathbb{R}^{N \times H \times r}, \quad q_{\text{pass}} \in \mathbb{R}^{N \times H \times (D-r)}
$$

<span style="font-size: 14px;">where $r = \texttt{rope\_dim}$. The rotated half is updated using the rotate-half form of RoPE:</span>

$$
q'_{\text{rope}} = q_{\text{rope}} \odot \cos_{\text{full}} + \text{rotate\_half}(q_{\text{rope}}) \odot \sin_{\text{full}}
$$

<span style="font-size: 14px;">The output is concatenation:</span>

$$
q' = [\, q'_{\text{rope}} \;\|\; q_{\text{pass}} \,]
$$

<span style="font-size: 14px;">The same formula applies to $k$. The helper $\text{rotate\_half}$ splits its argument into two equal halves and swaps them with a sign flip:</span>

$$
\text{rotate\_half}([\, x_1 \;\|\; x_2 \,]) = [\, -x_2 \;\|\; x_1 \,]
$$

<span style="font-size: 14px;">The $\cos$ and $\sin$ tables are precomputed for each position. Their stored shape is $(N, r/2)$ because they hold half the rotary slice (one value per frequency, not per channel). They are tiled to length $r$ before multiplication:</span>

$$
\cos_{\text{full}} = [\, \cos \;\|\; \cos \,], \qquad \sin_{\text{full}} = [\, \sin \;\|\; \sin \,]
$$

<span style="font-size: 14px;">After tiling, an extra axis is inserted at position 1 to broadcast across the $H$ head dimension. Final broadcast shape: $(N, 1, r)$.</span>

---

## <span style="font-size: 16px;">Rotate-Half vs Interleaved-Pair</span>

<span style="font-size: 14px;">Two equivalent formulations of RoPE appear in the literature. They differ only in how channels are paired:</span>

* <span style="font-size: 14px;">**Interleaved pairs (original paper, GPT-J):** pair channel $2i$ with channel $2i + 1$. The rotation acts on each pair independently.</span>
* <span style="font-size: 14px;">**Rotate-half (HuggingFace, GLM, LLaMA):** pair channel $i$ with channel $i + r/2$. The first half rotates against the second half.</span>

<span style="font-size: 14px;">Both yield the same relative-position dot product because they are related by a fixed permutation of channels. The rotate-half form is faster on accelerators since it slices contiguous halves instead of strided pairs. Partial RoPE in GLM-4.5 uses the rotate-half form, and the $\cos$/$\sin$ tables follow the same convention.</span>

---

## <span style="font-size: 16px;">Worked Numerical Example</span>

<span style="font-size: 14px;">Take $N = 1$ token, $H = 1$ head, $D = 4$, $r = 2$. So one channel pair gets rotated and two channels pass through.</span>

<span style="font-size: 14px;">Let $q = [0.5, 1.0, 2.0, 3.0]$ and $k = [0.2, 0.4, 0.6, 0.8]$. Suppose $\cos = [0.8]$ and $\sin = [0.6]$ (one frequency value, so $\cos$, $\sin$ have shape $(1, 1)$).</span>

<span style="font-size: 14px;">1. **Split.** $q_{\text{rope}} = [0.5, 1.0]$, $q_{\text{pass}} = [2.0, 3.0]$. Similarly $k_{\text{rope}} = [0.2, 0.4]$, $k_{\text{pass}} = [0.6, 0.8]$.</span>

<span style="font-size: 14px;">2. **Tile cos and sin.** $\cos_{\text{full}} = [0.8, 0.8]$, $\sin_{\text{full}} = [0.6, 0.6]$.</span>

<span style="font-size: 14px;">3. **rotate_half on $q_{\text{rope}}$.** $\text{rotate\_half}([0.5, 1.0]) = [-1.0, 0.5]$.</span>

<span style="font-size: 14px;">4. **Apply.** $q'_{\text{rope}} = [0.5, 1.0] \cdot [0.8, 0.8] + [-1.0, 0.5] \cdot [0.6, 0.6] = [0.4, 0.8] + [-0.6, 0.3] = [-0.2, 1.1]$.</span>

<span style="font-size: 14px;">5. **Concat.** $q' = [-0.2, 1.1, 2.0, 3.0]$. The last two channels are untouched.</span>

<span style="font-size: 14px;">6. **Same for $k$.** $\text{rotate\_half}([0.2, 0.4]) = [-0.4, 0.2]$. $k'_{\text{rope}} = [0.16, 0.32] + [-0.24, 0.12] = [-0.08, 0.44]$. So $k' = [-0.08, 0.44, 0.6, 0.8]$.</span>

---

## <span style="font-size: 16px;">Partial RoPE vs Full RoPE vs NoPE</span>

* <span style="font-size: 14px;">**Full RoPE (LLaMA, Mistral, Qwen):** $r = D$. Every channel gets a phase. Highest position-sensitivity but high-frequency channels lose semantic capacity.</span>
* <span style="font-size: 14px;">**Partial RoPE (GLM-4.5, GPT-NeoX, Phi-3):** $0 < r < D$. Best of both: positional information in the low-frequency band, semantic capacity in the pass-through band.</span>
* <span style="font-size: 14px;">**NoPE (No Positional Encoding):** $r = 0$. Surprisingly competitive in causal LMs because the causal mask already breaks permutation symmetry. Used in some long-context experiments.</span>

<span style="font-size: 14px;">Concrete reasons GLM-4.5 chooses partial over full:</span>

* <span style="font-size: 14px;">**Longer effective wavelengths.** With $r < D$, the lowest base frequency $\theta^{-2(r/2 - 1)/r}$ corresponds to a longer wavelength than a full RoPE with the same $\theta$. This helps the model attend across larger token gaps.</span>
* <span style="font-size: 14px;">**Reduced high-frequency aliasing at extrapolation.** Fewer fast-oscillating channels means fewer channels that flip near-orthogonal between positions far outside training length.</span>
* <span style="font-size: 14px;">**Per-head linear residual.** The pass-through slice acts like a positional-bias-free residual that survives the dot product unchanged, which the model can use for token-identity matching.</span>

<span style="font-size: 14px;">The dot product $q^\top k$ between two rotated query and key vectors decomposes into two independent additive terms:</span>

$$
q'^\top k' = q'_{\text{rope}}{}^\top k'_{\text{rope}} + q_{\text{pass}}^\top k_{\text{pass}}
$$

<span style="font-size: 14px;">The first term encodes relative position via the standard RoPE identity $q'_{\text{rope}}{}^\top k'_{\text{rope}} = q_{\text{rope}}^\top R(\Delta m) k_{\text{rope}}$ where $\Delta m$ is the position offset between the two tokens. The second term is position-independent. The model gets to learn how much of each head to spend on each role: more position bandwidth means more channels in the rotated slice, more content bandwidth means more channels in the pass-through slice.</span>

---

## <span style="font-size: 16px;">Frequency Construction</span>

<span style="font-size: 14px;">The $\cos$ and $\sin$ tables are derived from a base $\theta$ (commonly $10000$ or larger for long-context models) and the rotary dimension $r$. The $i$-th inverse frequency is:</span>

$$
\omega_i = \frac{1}{\theta^{2i/r}}, \qquad i = 0, 1, \ldots, r/2 - 1
$$

<span style="font-size: 14px;">For token at position $m$, the angle for channel pair $i$ is $m \cdot \omega_i$, and the entry stored in the cos/sin tables at row $m$, column $i$ is $\cos(m \omega_i)$ or $\sin(m \omega_i)$. Important properties:</span>

* <span style="font-size: 14px;">**Lowest frequency channel pair ($i = 0$):** angular speed $1$. Wavelength is $2\pi$ tokens. This is the fastest rotation.</span>
* <span style="font-size: 14px;">**Highest frequency channel pair ($i = r/2 - 1$):** angular speed $1 / \theta^{(r-2)/r}$. Wavelength approaches $2\pi \theta$ for large $r$. This is the slowest rotation, used to encode coarse position over very long ranges.</span>
* <span style="font-size: 14px;">**Effect of shrinking $r$.** With a smaller rotary slice, the spacing of frequencies stretches. The slowest channel has wavelength close to $2\pi \theta^{(r-2)/r}$, which for moderate $r$ still spans tens of thousands of tokens.</span>

<span style="font-size: 14px;">GLM-4.5 uses a large $\theta$ to keep the slowest rotary channel's wavelength bigger than the training context window. This ensures every relative position within context falls within the first rotation period of the slowest channel.</span>

---

## <span style="font-size: 16px;">Implementation Notes for GLM-4.5</span>

* <span style="font-size: 14px;">**Input shape contract.** The reference uses $q$, $k$ of shape $(N, H, D)$. The $\cos$ and $\sin$ tables have shape $(N, r/2)$. The expanded tables broadcast to $(N, 1, r)$ before multiplication.</span>
* <span style="font-size: 14px;">**Same tables for $q$ and $k$.** Both share the position-conditioned $\cos$ and $\sin$. The rotation must be applied identically so relative position emerges from the dot product $q^\top k$.</span>
* <span style="font-size: 14px;">**Order of operations.** Slice first, then rotate, then concatenate. Reversing this (concat then rotate) wastes compute and risks zeroing the pass-through.</span>
* <span style="font-size: 14px;">**Dtype.** RoPE is typically done in float32 even when the model runs in bf16, because the angle precision matters. The reference here uses float64 for test determinism.</span>
* <span style="font-size: 14px;">**Caching.** The $\cos$ and $\sin$ tables are precomputed once per layer (or shared across layers) up to the max sequence length. Re-computing per step is the leading cause of slow long-context inference.</span>

---

## <span style="font-size: 16px;">Complexity and Memory</span>

* <span style="font-size: 14px;">**Compute.** Partial RoPE costs one elementwise multiply and one rotate-half plus another multiply, restricted to the rotated slice. Total floating-point operations scale as $O(N \cdot H \cdot r)$, strictly cheaper than full RoPE's $O(N \cdot H \cdot D)$.</span>
* <span style="font-size: 14px;">**Memory.** The $\cos$ and $\sin$ tables are shape $(N, r/2)$ each, so memory grows linearly in context length and the rotary dimension. For typical configurations with $r = D/2$ this halves the trig table size relative to full RoPE.</span>
* <span style="font-size: 14px;">**Cache reuse.** The same $\cos$/$\sin$ pair is reused across all heads, all layers, and both $q$ and $k$. Building them once and sharing is the standard optimisation.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Rotating all channels.** The most common bug: applying RoPE to the entire head instead of slicing first. The result still has the right shape and looks plausible on small inputs but the model has no untouched residual slice.</span>
* <span style="font-size: 14px;">**Dropping or zeroing the pass-through.** Replacing $q_{\text{pass}}$ with zeros (or scaling it) destroys the semantic content that the model expects to carry through. This silently degrades quality without crashing.</span>
* <span style="font-size: 14px;">**Wrong sign in rotate-half.** The rotate-half convention is $[-x_2, x_1]$. Writing $[x_2, -x_1]$ rotates by the opposite angle, which corresponds to the conjugate frequency. Dot products with $k$ then encode the negative of the intended relative position.</span>
* <span style="font-size: 14px;">**Swapping $\cos$ and $\sin$.** Treating $\sin$ as the multiplier on the identity term and $\cos$ as the multiplier on $\text{rotate\_half}$ produces a rotation by $\pi/2 - \theta$ rather than $\theta$, breaking relative-position symmetry.</span>
* <span style="font-size: 14px;">**Forgetting to apply RoPE to $k$.** The relative-position property of RoPE depends on rotating both $q$ and $k$. If only $q$ is rotated, the dot product becomes position-dependent in an asymmetric way that the model was not trained on.</span>
* <span style="font-size: 14px;">**Skipping the unsqueeze for the head axis.** $\cos$ and $\sin$ have shape $(N, r/2)$. After concat they become $(N, r)$. They must be reshaped to $(N, 1, r)$ to broadcast across $H$ heads. Forgetting the unsqueeze either errors or silently broadcasts wrong.</span>
* <span style="font-size: 14px;">**Mixing rotate-half and interleaved-pair conventions.** If the $\cos$/$\sin$ tables were precomputed for interleaved pairs (paired channels $2i$ and $2i+1$) but the code uses rotate-half (paired channels $i$ and $i + r/2$), the rotation matrices act on the wrong subspaces and the model produces garbage.</span>
* <span style="font-size: 14px;">**Wrong $\texttt{rope\_dim}$ parity.** $\texttt{rope\_dim}$ must be even because the rotation pairs channels in halves of size $r/2$. An odd value either errors or silently truncates.</span>
* <span style="font-size: 14px;">**In-place mutation.** Writing into $q$ instead of returning a new tensor breaks gradient flow if $q$ was an intermediate activation. Always return new tensors.</span>

---

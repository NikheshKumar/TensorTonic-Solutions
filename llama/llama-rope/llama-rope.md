# <span style="font-size: 20px;">Rotary Positional Embeddings (RoPE)</span>

<span style="font-size: 14px;">Rotary Positional Embedding (RoPE) encodes position information by rotating query and key vectors in the attention mechanism. Instead of adding a positional vector to token embeddings, RoPE applies a position-dependent rotation to each consecutive pair of dimensions in Q and K, so that the dot product between any two positions naturally depends only on their relative distance. Introduced by Su et al. (2021) in the RoFormer paper, RoPE was adopted by LLaMA, Mistral, Gemma, and most modern open-weight LLMs.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">Positional embeddings solve a fundamental problem in Transformers: self-attention is permutation-invariant. Without position information, the sentence "the cat sat on the mat" and "mat the on sat cat the" produce identical attention patterns. The model needs some mechanism to distinguish token order.</span>

<span style="font-size: 14px;">RoPE takes a different approach from earlier methods. Instead of adding a positional signal to the input embeddings, RoPE modifies the query and key vectors directly inside the attention computation. It applies a rotation whose angle depends on the token's position. The rotation is applied to consecutive pairs of dimensions: dimensions $(0, 1)$ form one pair, $(2, 3)$ form another, and so on. Each pair is rotated by a different angle, where the angle is the product of the position index and a frequency specific to that pair.</span>

<span style="font-size: 14px;">The critical property is that after rotation, the dot product $q_m^T k_n$ between a query at position $m$ and a key at position $n$ depends only on the relative distance $m - n$, not on the absolute positions individually. A word three positions to the left always "looks" the same regardless of whether it sits at positions $(5, 2)$ or $(100, 97)$.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">**The rotation formula.** Given a vector $x \in \mathbb{R}^d$ at position $m$, RoPE modifies each consecutive pair of dimensions $(2i, 2i+1)$ as follows:</span>

$$
x'_{2i} = x_{2i} \cos(m \theta_i) - x_{2i+1} \sin(m \theta_i)
$$

$$
x'_{2i+1} = x_{2i} \sin(m \theta_i) + x_{2i+1} \cos(m \theta_i)
$$

<span style="font-size: 14px;">where $i = 0, 1, \dots, d/2 - 1$ indexes the pair, $m$ is the position in the sequence, and $\theta_i$ is the frequency for pair $i$.</span>

<span style="font-size: 14px;">**The frequency schedule.** Each pair has a distinct frequency:</span>

$$
\theta_i = \frac{1}{10000^{2i/d}}
$$

<span style="font-size: 14px;">Low-index pairs have high frequencies (rapidly changing angles), while high-index pairs have low frequencies (slowly changing angles). The base value 10000 is a hyperparameter; LLaMA uses 10000, while some models use larger bases for longer context support.</span>

<span style="font-size: 14px;">**2D rotation matrix form.** The operation on each pair $(x_{2i}, x_{2i+1})$ is exactly a 2D rotation matrix applied to a 2-element vector:</span>

$$
\begin{pmatrix} x'_{2i} \\ x'_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}
$$

<span style="font-size: 14px;">The full $d$-dimensional rotation is a block-diagonal matrix $R_m$ with $d/2$ independent $2 \times 2$ rotation blocks. The rotated query and key are $q'_m = R_m \, q_m$ and $k'_n = R_n \, k_n$.</span>

<span style="font-size: 14px;">**How the dot product encodes relative position.** The attention score between positions $m$ and $n$ is:</span>

$$
q'^T_m k'_n = (R_m \, q_m)^T (R_n \, k_n) = q_m^T R_m^T R_n \, k_n = q_m^T R_{n-m} \, k_n
$$

<span style="font-size: 14px;">The last step uses the fact that transposing a rotation reverses its angle, and composing rotations adds their angles: $(-m\theta_i) + (n\theta_i) = (n - m)\theta_i$. The result depends only on the relative position $n - m$.</span>

---

## <span style="font-size: 16px;">Why Rotation Encodes Position</span>

<span style="font-size: 14px;">The key insight is that 2D rotations compose additively in their angles. If you rotate a vector by angle $\alpha$ and then by angle $\beta$, the net rotation is $\alpha + \beta$. This additive property is what makes relative position encoding fall out naturally.</span>

<span style="font-size: 14px;">**Proof sketch.** Consider a single pair. Let $q = (q_0, q_1)$ at position $m$ and $k = (k_0, k_1)$ at position $n$. After applying RoPE, the contribution of this pair to the dot product is:</span>

$$
q'_0 k'_0 + q'_1 k'_1
$$

<span style="font-size: 14px;">Expanding using the rotation formulas and distributing, the cross terms cancel via the identity $\cos(m\theta)\cos(n\theta) + \sin(m\theta)\sin(n\theta) = \cos((m-n)\theta)$. The result simplifies to:</span>

$$
q_0 k_0 \cos((m-n)\theta) - q_0 k_1 \sin((m-n)\theta) + q_1 k_0 \sin((m-n)\theta) + q_1 k_1 \cos((m-n)\theta)
$$

<span style="font-size: 14px;">This is exactly $(q_0, q_1)^T R_{m-n} (k_0, k_1)$, confirming the dot product depends only on the relative position $m - n$. This trigonometric cancellation is the mathematical core of why rotation works as a relative positional encoding.</span>

<span style="font-size: 14px;">**Geometric interpretation.** Think of each pair of dimensions as a 2D plane. The token content determines a vector in this plane. RoPE rotates this vector by an angle proportional to position. When computing the dot product, two vectors rotated by similar amounts will be well-aligned, while vectors rotated by very different amounts will be misaligned. The dot product measures how much the rotation difference (the positional gap) affects alignment, which is exactly relative position information.</span>

---

## <span style="font-size: 16px;">The Pair-Wise Operation</span>

<span style="font-size: 14px;">RoPE splits the head dimension $d_{\text{head}}$ into $d_{\text{head}} / 2$ pairs. Each pair consists of two adjacent dimensions and is treated as a 2D vector that gets rotated independently. In LLaMA with $d_{\text{head}} = 128$, this creates 64 independent pairs.</span>

<span style="font-size: 14px;">**Different frequencies per pair.** Pair $i$ is rotated at frequency $\theta_i = 1 / 10000^{2i/d}$. The first pair ($i = 0$) has $\theta_0 = 1.0$, meaning the angle changes by 1 radian per position, capturing fine-grained local positional information. The last pair has an extremely small $\theta$, barely changing between adjacent positions, capturing long-range coarse positional structure.</span>

<span style="font-size: 14px;">**Analogy to Fourier decomposition.** The multi-frequency structure is analogous to a Fourier decomposition. High-frequency pairs distinguish nearby positions (5 vs 6), while low-frequency pairs distinguish distant positions (5 vs 500). Together, the $d/2$ pairs create a rich, multi-scale encoding of position.</span>

<span style="font-size: 14px;">**Independence of pairs.** Because the rotation matrix $R_m$ is block-diagonal, each pair is rotated without interaction with other pairs. This makes the operation embarrassingly parallel. The total cost is equivalent to an element-wise multiply plus an element-wise add, since the rotation can be implemented without constructing an explicit matrix.</span>

---

## <span style="font-size: 16px;">Applied to Q and K Only</span>

<span style="font-size: 14px;">RoPE is applied only to the query and key vectors, not to the value vectors or any other part of the Transformer. This is a deliberate design choice.</span>

<span style="font-size: 14px;">**Why Q and K?** The attention mechanism computes weights via $\text{softmax}(Q K^T / \sqrt{d_k})$ and applies those weights to V. Position information should affect which tokens attend to which (the attention pattern), not the content being aggregated. By rotating only Q and K, RoPE injects position into the attention weight computation while leaving value content position-free.</span>

<span style="font-size: 14px;">**Not added, multiplied.** Unlike sinusoidal or learned positional embeddings that are added to input embeddings before the attention layer, RoPE is multiplicative. It rotates Q and K after they have been projected from the hidden states. This means the positional encoding is applied at every layer independently, not just once at the input, giving each attention layer a fresh positional signal.</span>

<span style="font-size: 14px;">**Implementation detail.** In LLaMA, the forward pass proceeds as: (1) compute Q, K, V via linear projections, (2) apply RoPE rotation to Q and K, (3) compute attention scores from rotated Q and K, (4) apply attention weights to unrotated V. The rotation happens between projection and attention, making it a lightweight modification to the standard pipeline.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">**Su et al. (2021) -- "RoFormer: Enhanced Transformer with Rotary Position Embedding."** This paper introduced RoPE as a method that unifies absolute and relative positional encoding. The authors showed that formulating position as a rotation makes the dot product naturally encode relative position without any explicit relative bias term. The paper demonstrated improvements over both learned absolute embeddings and relative methods like T5's relative bias.</span>

<span style="font-size: 14px;">**Touvron et al. (2023) -- "LLaMA: Open and Efficient Foundation Language Models."** LLaMA adopted RoPE alongside RMSNorm and SwiGLU activations. The paper cited RoPE's ability to handle variable sequence lengths. LLaMA's success popularized RoPE, and virtually every subsequent open-weight LLM (LLaMA 2, Mistral, Gemma, Qwen, DeepSeek) adopted it.</span>

<span style="font-size: 14px;">**Context length extensions.** RoPE's design has enabled several context extension techniques. Position Interpolation (Chen et al., 2023) scales down position indices to fit within the trained range. Code LLaMA extended the base from 10000 to 1000000. YaRN combines interpolation with frequency-dependent scaling. These methods work because RoPE's rotation-based encoding is mathematically structured and amenable to principled modifications.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider $d_{\text{head}} = 4$, giving $d/2 = 2$ pairs. Let the query vector at position $m = 3$ be $q = [1.0, 2.0, 3.0, 4.0]$.</span>

<span style="font-size: 14px;">**Step 1 -- Compute the frequencies.** With $d = 4$:</span>

$$
\theta_0 = \frac{1}{10000^{0/4}} = 1.0, \quad \theta_1 = \frac{1}{10000^{2/4}} = \frac{1}{100} = 0.01
$$

<span style="font-size: 14px;">**Step 2 -- Compute the rotation angles.** At position $m = 3$:</span>

$$
\alpha_0 = 3 \times 1.0 = 3.0 \text{ rad}, \quad \alpha_1 = 3 \times 0.01 = 0.03 \text{ rad}
$$

<span style="font-size: 14px;">**Step 3 -- Compute cosines and sines:**</span>

$$
\cos(3.0) \approx -0.9900, \quad \sin(3.0) \approx 0.1411
$$

$$
\cos(0.03) \approx 0.9996, \quad \sin(0.03) \approx 0.0300
$$

<span style="font-size: 14px;">**Step 4 -- Rotate pair 0: dimensions (0, 1).** Input: $(q_0, q_1) = (1.0, 2.0)$.</span>

$$
q'_0 = 1.0 \times (-0.9900) - 2.0 \times 0.1411 = -0.9900 - 0.2822 = -1.2722
$$

$$
q'_1 = 1.0 \times 0.1411 + 2.0 \times (-0.9900) = 0.1411 - 1.9800 = -1.8389
$$

<span style="font-size: 14px;">**Step 5 -- Rotate pair 1: dimensions (2, 3).** Input: $(q_2, q_3) = (3.0, 4.0)$.</span>

$$
q'_2 = 3.0 \times 0.9996 - 4.0 \times 0.0300 = 2.9988 - 0.1200 = 2.8788
$$

$$
q'_3 = 3.0 \times 0.0300 + 4.0 \times 0.9996 = 0.0900 + 3.9984 = 4.0884
$$

<span style="font-size: 14px;">**Result.** The rotated query is $q' = [-1.2722, -1.8389, 2.8788, 4.0884]$.</span>

<span style="font-size: 14px;">**Observations.** Pair 0 was rotated by 3.0 radians (nearly $\pi$), causing a near-reversal of direction. Pair 1 was rotated by only 0.03 radians, barely changing the vector. This illustrates the multi-scale nature: high-frequency pairs undergo large rotations and are sensitive to exact position, while low-frequency pairs change slowly and encode coarse position information.</span>

---

## <span style="font-size: 16px;">RoPE vs Learned vs Sinusoidal</span>

<span style="font-size: 14px;">**Learned positional embeddings.** Used by GPT-2 and BERT. A learnable matrix $P \in \mathbb{R}^{L_{\max} \times d}$ maps each absolute position to a vector added to the token embedding. The main limitation is that $L_{\max}$ is fixed at training time, and the model cannot generalize beyond it. Learned embeddings also encode absolute position, so the model must independently learn that "token 3 attending to token 1" and "token 103 attending to token 101" represent the same relationship.</span>

<span style="font-size: 14px;">**Sinusoidal positional embeddings.** Used by the original Transformer. Position is encoded as sines and cosines at different frequencies, added to the input. The dot product of two sinusoidal embeddings depends on position difference, but because the vector is added before the Q/K projections, the relative position signal degrades after passing through the linear projections.</span>

<span style="font-size: 14px;">**RoPE.** By applying rotation after the Q/K projections, RoPE ensures the relative position property holds in the actual attention score, not just in raw embeddings. RoPE has no fixed maximum length, introduces no learned parameters, and is compatible with context extension methods.</span>

* <span style="font-size: 14px;">**Learned:** Flexible within training length, but fixed $L_{\max}$, absolute-only, extra parameters.</span>
* <span style="font-size: 14px;">**Sinusoidal:** No learned parameters, theoretically relative, but additive application dilutes the signal through projections.</span>
* <span style="font-size: 14px;">**RoPE:** No learned parameters, true relative position in attention scores, extrapolatable, dominant in modern LLMs.</span>

---

## <span style="font-size: 16px;">Pitfalls and Common Mistakes</span>

* <span style="font-size: 14px;">**Rotating V vectors.** RoPE must only be applied to Q and K. Applying it to V injects position information into the value content, distorting the semantic representation. The attention weights already carry position information; values should remain position-free.</span>

* <span style="font-size: 14px;">**Wrong pair grouping.** RoPE pairs consecutive dimensions: $(0, 1)$, $(2, 3)$, $(4, 5)$, etc. Some implementations use a "non-interleaved" layout that pairs the first half with the second half: $(0, d/2)$, $(1, d/2+1)$, etc. Either convention works, but it must be consistent between Q and K. Mixing conventions produces incorrect attention scores.</span>

* <span style="font-size: 14px;">**Applying the same theta to all pairs.** Each pair must use a different frequency $\theta_i$. Using a single frequency collapses the multi-scale structure and severely degrades position sensitivity.</span>

* <span style="font-size: 14px;">**Forgetting to apply to both Q and K.** The relative position property requires rotation of both vectors. If only Q is rotated, the dot product becomes $q_m^T R_m k_n$, which depends on absolute position $m$ rather than relative position $m - n$.</span>

* <span style="font-size: 14px;">**Rotation direction.** Both Q and K are rotated with positive angles ($+m\theta$ and $+n\theta$). The transpose in $Q K^T$ produces the negation that yields $(m - n)\theta$. If you accidentally negate the angle for one of Q or K, the result becomes $(m + n)\theta$, encoding the sum of positions rather than the difference.</span>

* <span style="font-size: 14px;">**Applying RoPE before projection.** RoPE must be applied after the Q and K linear projections, not before. Applying it to the hidden state before projection causes the linear layer to mix the rotated dimensions, destroying the pair-wise rotation structure.</span>

---
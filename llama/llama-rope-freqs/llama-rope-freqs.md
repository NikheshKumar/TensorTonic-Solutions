# <span style="font-size: 20px;">RoPE Frequency Table</span>

<span style="font-size: 14px;">Rotary Position Embedding (RoPE) encodes token positions by rotating query and key vectors in the attention mechanism. Before any forward pass begins, LLaMA precomputes two tables of cosine and sine values -- one entry for every combination of sequence position and dimension pair. This problem asks you to build exactly those tables: the foundation on which every attention layer in LLaMA relies for positional awareness.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">The RoPE frequency table is a pair of precomputed matrices, one for cosine values and one for sine values, each of shape $(L, d/2)$ where $L$ is the maximum sequence length and $d$ is the head dimension. Each entry stores the cosine or sine of an angle that depends on two things: the absolute position of a token in the sequence, and which dimension pair the rotation acts on.</span>

<span style="font-size: 14px;">In the original Transformer (Vaswani et al., 2017), sinusoidal encodings are added to embeddings at the input layer. RoPE takes a different approach: instead of adding positional information, it multiplies query and key vectors by rotation matrices. Each 2D sub-plane of the head dimension receives its own rotation whose angle depends on token position. The frequency table stores all rotation angles (as cos and sin values) so they never need to be recomputed during training or inference.</span>

<span style="font-size: 14px;">The table is static -- it depends only on hyperparameters (base frequency, head dimension, maximum sequence length) and never changes during training. Every attention layer in every block of LLaMA shares the same precomputed table, computed once at model initialization.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">The construction proceeds in three stages: compute the base frequencies, form the full angle matrix, then take cosines and sines.</span>

<span style="font-size: 14px;">**Stage 1 -- Base frequencies.** For a head dimension $d$ and a base value $b$, define the frequency for dimension pair $i$ (where $i = 0, 1, \ldots, d/2 - 1$) as:</span>

$$
\theta_i = \frac{1}{b^{\,2i\,/\,d}}
$$

<span style="font-size: 14px;">This produces $d/2$ frequency values. The exponent $2i/d$ ranges from $0$ (for $i=0$) to $(d-2)/d$ (for $i = d/2 - 1$). Since $b$ is a large number (10,000 or 500,000), raising it to a fractional power and taking the reciprocal produces frequencies that decrease geometrically from $\theta_0 = 1.0$ down to a very small value for the last dimension pair.</span>

<span style="font-size: 14px;">**Stage 2 -- Angle matrix via outer product.** Let $\mathbf{p} = [0, 1, 2, \ldots, L-1]$ be the vector of all positions and $\boldsymbol{\theta} = [\theta_0, \theta_1, \ldots, \theta_{d/2-1}]$ be the vector of all frequencies. The full angle matrix $\mathbf{A}$ is their outer product:</span>

$$
\mathbf{A} = \mathbf{p} \otimes \boldsymbol{\theta} \quad \in \mathbb{R}^{L \times d/2}
$$

<span style="font-size: 14px;">Each entry is simply:</span>

$$
A[p, i] = p \cdot \theta_i
$$

<span style="font-size: 14px;">Position $p = 0$ produces a row of all zeros (no rotation). Position $p = 1$ produces a row equal to the frequency vector itself. Position $p = 100$ produces angles that are 100 times the base frequencies.</span>

<span style="font-size: 14px;">**Stage 3 -- Cosine and sine tables.** Apply element-wise cosine and sine to the angle matrix:</span>

$$
\text{freqs\_cos}[p, i] = \cos(A[p, i]) = \cos(p \cdot \theta_i)
$$

$$
\text{freqs\_sin}[p, i] = \sin(A[p, i]) = \sin(p \cdot \theta_i)
$$

<span style="font-size: 14px;">Both tables have shape $(L, d/2)$. During the attention computation, these tables rotate each pair of consecutive dimensions in the query and key vectors. For a query vector $\mathbf{q}$ at position $p$, the rotation of dimension pair $i$ is:</span>

$$
\begin{bmatrix} q'_{2i} \\ q'_{2i+1} \end{bmatrix} = \begin{bmatrix} \cos(p \cdot \theta_i) & -\sin(p \cdot \theta_i) \\ \sin(p \cdot \theta_i) & \cos(p \cdot \theta_i) \end{bmatrix} \begin{bmatrix} q_{2i} \\ q_{2i+1} \end{bmatrix}
$$

<span style="font-size: 14px;">This is a standard 2D rotation matrix applied independently to each pair. The cosine and sine tables provide exactly the values needed for this operation across all positions and all dimension pairs.</span>

---

## <span style="font-size: 16px;">Why Precompute</span>

<span style="font-size: 14px;">**Shared across all layers and heads.** In LLaMA, every attention layer applies the same rotary embedding to queries and keys. LLaMA 7B has 32 layers with 32 heads each; LLaMA 3 405B has 126 layers. Every layer and head uses the exact same cos/sin values for a given position and dimension pair. Without precomputation, each layer would redundantly compute transcendental functions on every forward pass.</span>

<span style="font-size: 14px;">**No learned parameters.** Unlike learned positional embeddings (GPT-2, BERT), RoPE's frequency table has no trainable weights. It is deterministic given the hyperparameters, computed once at initialization and stored as a buffer. During backpropagation, gradients flow through the rotation to queries and keys, but the cos/sin values are treated as constants.</span>

<span style="font-size: 14px;">**Cost of transcendental functions.** Computing $\cos$ and $\sin$ is far more expensive than a multiply or add. For sequence length 8192 with head dimension 128, the table has $8192 \times 64 = 524{,}288$ entries. Computing this once is negligible; repeating it at every layer of every forward pass would add measurable overhead.</span>

---

## <span style="font-size: 16px;">The Base Frequency</span>

<span style="font-size: 14px;">The base frequency $b$ is the single most important hyperparameter in the RoPE frequency table. It controls how quickly the rotation frequencies decay across dimensions, which in turn controls the range of positional information each dimension pair captures.</span>

<span style="font-size: 14px;">**The original value: $b = 10{,}000$.** Su et al. (2021) chose $b = 10{,}000$, the same base used in the original Transformer's sinusoidal encoding. With this base and a typical head dimension of $d = 128$:</span>

* <span style="font-size: 14px;">**Dimension pair 0** ($i=0$): $\theta_0 = 1/10000^{0/128} = 1.0$. One full rotation every ~6.28 positions. Captures very short-range relationships.</span>
* <span style="font-size: 14px;">**Dimension pair 32** ($i=32$): $\theta_{32} = 1/10000^{0.5} = 0.01$. One full rotation every ~628 positions. Captures medium-range relationships.</span>
* <span style="font-size: 14px;">**Dimension pair 63** ($i=63$): $\theta_{63} = 1/10000^{126/128} \approx 0.000116$. One full rotation every ~54,000 positions. Captures extremely long-range relationships.</span>

<span style="font-size: 14px;">**LLaMA 3's value: $b = 500{,}000$.** Meta increased the base from $10{,}000$ to $500{,}000$ for LLaMA 3's 128K-token context (Grattafiori et al., 2024). A larger base makes all frequencies smaller, so rotations are slower and positions much farther apart remain distinguishable. For dimension pair 32 with $d = 128$: the frequency drops from $0.01$ to $1/\sqrt{500000} \approx 0.001414$, a ~7x decrease stretching the wavelength by the same factor.</span>

<span style="font-size: 14px;">**The wavelength analogy.** The base controls a spectrum of wavelengths. Low-indexed dimensions are high-frequency components (short wavelength, nearby positions); high-indexed dimensions are low-frequency components (long wavelength, distant positions). Increasing the base stretches all wavelengths toward longer ranges, enabling long-context modeling.</span>

---

## <span style="font-size: 16px;">How Frequencies Decay</span>

<span style="font-size: 14px;">The frequencies $\theta_i$ form a **geometric progression**. This follows directly from $\theta_i = 1/b^{2i/d}$. The ratio of consecutive frequencies is:</span>

$$
\frac{\theta_{i+1}}{\theta_i} = \frac{b^{-2(i+1)/d}}{b^{-2i/d}} = b^{-2/d}
$$

<span style="font-size: 14px;">This ratio is constant -- it does not depend on $i$. With $b = 10{,}000$ and $d = 128$, the common ratio is $10000^{-1/64} \approx 0.8617$. Each successive frequency is about 86% of the previous one.</span>

<span style="font-size: 14px;">**Low dimensions rotate fast.** The first dimension pairs have frequencies close to 1.0. At position $p = 100$, dimension pair 0 has angle $100$ radians, meaning $100/(2\pi) \approx 15.9$ full cycles. These dimensions are sensitive to small position differences but aliased for distant positions because the rotation wraps around many times.</span>

<span style="font-size: 14px;">**High dimensions rotate slowly.** The last dimension pair has a frequency near $1/b$. At position $p = 100$ with $b = 10{,}000$, the angle is just $0.01$ radians -- the rotation has barely moved. These dimensions change gradually with position, ideal for encoding coarse, long-range information.</span>

<span style="font-size: 14px;">**Multi-scale representation.** Together, the spectrum from fast to slow gives RoPE a multi-scale positional encoding. The attention mechanism simultaneously leverages short-range precision (fast-rotating dimensions) and long-range awareness (slow-rotating dimensions), analogous to how Fourier features at multiple frequencies represent complex patterns.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">**RoFormer (Su et al., 2021).** Rotary Position Embedding was introduced in "RoFormer: Enhanced Transformer with Rotary Position Embedding." The core insight: multiplying by rotation matrices -- rather than adding positional vectors -- naturally encodes relative position in the dot product. When a query at position $m$ and a key at position $n$ are both rotated, their dot product depends only on $m - n$. Su et al. showed that the rotated dot product decomposes into a sum over dimension pairs, each contributing a different frequency component of the relative position signal. The precomputed frequency table makes this efficient: look up cos/sin values, apply the rotation, and the relative-position property falls out automatically.</span>

<span style="font-size: 14px;">**LLaMA (Touvron et al., 2023).** The original LLaMA adopted RoPE with base $b = 10{,}000$ and context length 2048, replacing learned position embeddings. RoPE was chosen for good length extrapolation and zero trainable positional parameters.</span>

<span style="font-size: 14px;">**LLaMA 2 (Touvron et al., 2023).** Extended context to 4096 tokens while keeping $b = 10{,}000$. The frequency table simply added more rows -- no other RoPE changes needed, showing that extending context only requires extending the precomputed tables.</span>

<span style="font-size: 14px;">**LLaMA 3 (Grattafiori et al., 2024).** Increased the base to $b = 500{,}000$ for 128K-token context. With the original base, high-frequency dimensions would cycle too many times over 128K positions, losing the ability to distinguish distant tokens. The 50x increase stretched all wavelengths proportionally.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Let us work through a complete example with $d = 4$ (head dimension), $L = 3$ (max sequence length), and $b = 10{,}000$.</span>

<span style="font-size: 14px;">**Step 1 -- Compute the frequency vector.** With $d = 4$, we have $d/2 = 2$ dimension pairs, indexed $i = 0$ and $i = 1$.</span>

$$
\theta_0 = \frac{1}{10000^{\,2 \cdot 0 / 4}} = \frac{1}{10000^0} = 1.0
$$

$$
\theta_1 = \frac{1}{10000^{\,2 \cdot 1 / 4}} = \frac{1}{10000^{0.5}} = \frac{1}{100} = 0.01
$$

<span style="font-size: 14px;">The frequency vector is $\boldsymbol{\theta} = [1.0, 0.01]$. The first pair rotates 100 times faster than the second.</span>

<span style="font-size: 14px;">**Step 2 -- Form the angle matrix via outer product.** The position vector is $\mathbf{p} = [0, 1, 2]$. The outer product $\mathbf{A} = \mathbf{p} \otimes \boldsymbol{\theta}$ gives:</span>

$$
A[0, 0] = 0 \times 1.0 = 0.0 \qquad A[0, 1] = 0 \times 0.01 = 0.0
$$

$$
A[1, 0] = 1 \times 1.0 = 1.0 \qquad A[1, 1] = 1 \times 0.01 = 0.01
$$

$$
A[2, 0] = 2 \times 1.0 = 2.0 \qquad A[2, 1] = 2 \times 0.01 = 0.02
$$

<span style="font-size: 14px;">The full angle matrix is:</span>

$$
\mathbf{A} = \begin{bmatrix} 0.0 & 0.0 \\ 1.0 & 0.01 \\ 2.0 & 0.02 \end{bmatrix}
$$

<span style="font-size: 14px;">**Step 3 -- Compute the cosine table.** Apply $\cos$ element-wise:</span>

$$
\cos(0.0) = 1.0000 \qquad \cos(0.0) = 1.0000
$$

$$
\cos(1.0) = 0.5403 \qquad \cos(0.01) = 0.99995
$$

$$
\cos(2.0) = -0.4161 \qquad \cos(0.02) = 0.9998
$$

<span style="font-size: 14px;">The cosine table:</span>

$$
\text{freqs\_cos} = \begin{bmatrix} 1.0000 & 1.0000 \\ 0.5403 & 0.99995 \\ -0.4161 & 0.9998 \end{bmatrix}
$$

<span style="font-size: 14px;">**Step 4 -- Compute the sine table.** Apply $\sin$ element-wise:</span>

$$
\sin(0.0) = 0.0000 \qquad \sin(0.0) = 0.0000
$$

$$
\sin(1.0) = 0.8415 \qquad \sin(0.01) = 0.01000
$$

$$
\sin(2.0) = 0.9093 \qquad \sin(0.02) = 0.02000
$$

<span style="font-size: 14px;">The sine table:</span>

$$
\text{freqs\_sin} = \begin{bmatrix} 0.0000 & 0.0000 \\ 0.8415 & 0.01000 \\ 0.9093 & 0.02000 \end{bmatrix}
$$

<span style="font-size: 14px;">**Observations:**</span>

* <span style="font-size: 14px;">**Row 0 (position 0):** All angles are zero, so $\cos = 1$ and $\sin = 0$ everywhere. The token at position 0 receives no rotation -- this is the identity rotation.</span>
* <span style="font-size: 14px;">**Column 0 (fast dimension):** Cosine swings from 1.0 to 0.54 to -0.42 in just three positions. Sine ramps from 0 to 0.84 to 0.91. The rotation is happening rapidly.</span>
* <span style="font-size: 14px;">**Column 1 (slow dimension):** Cosine barely moves from 1.0 (still 0.9998 at position 2). Sine barely moves from 0.0 (only 0.02 at position 2). This dimension pair is nearly static over short distances -- it would take hundreds of positions to see significant rotation.</span>
* <span style="font-size: 14px;">**Multi-scale behavior:** Even with just two dimension pairs, the table exhibits the core RoPE property: one dimension captures fine-grained token-by-token differences, while the other captures coarse long-range information.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

<span style="font-size: 14px;">**Wrong exponent denominator.** The most common mistake is using $2i / d_{\text{model}}$ instead of $2i / d_{\text{head}}$. RoPE operates on individual attention heads. If $d_{\text{model}} = 4096$ and $d_{\text{head}} = 128$, the wrong denominator produces 2048 frequencies instead of 64, and all exponents are 32x smaller than intended, yielding frequencies far too close to 1.0. Always use $d_{\text{head}}$ in $\theta_i = 1/b^{2i/d_{\text{head}}}$.</span>

<span style="font-size: 14px;">**Forgetting that the output has half the head dimension.** Each rotation acts on a pair of dimensions $(2i, 2i+1)$, so $d$ dimensions produce $d/2$ frequency values. The output tables have shape $(L, d/2)$, not $(L, d)$. The loop over $i$ should run from $0$ to $d/2 - 1$, producing exactly $d/2$ entries per position.</span>

<span style="font-size: 14px;">**Using the wrong base value.** LLaMA 1/2 use $b = 10{,}000$. LLaMA 3 uses $b = 500{,}000$. Using one when the other is expected produces entirely different frequency tables. Always check which base is specified -- the difference fundamentally changes the encoding's range and resolution.</span>

<span style="font-size: 14px;">**Integer division in the exponent.** In Python 3, `/` gives a float, so `2*i / d` works correctly. But in C/C++/Java, integer division truncates: `2*i / d` with `i=1, d=128` gives `0` instead of `0.015625`. This collapses every frequency to $1/b^0 = 1.0$, destroying the multi-scale structure. Always ensure floating-point division.</span>

<span style="font-size: 14px;">**Off-by-one in the dimension index.** The exponent uses $2i$ where $i$ ranges from $0$ to $d/2 - 1$. Some implementations index from $1$ to $d/2$, shifting all frequencies. Others omit the factor of 2, producing exponents $0/d, 1/d, 2/d, \ldots$ instead of $0/d, 2/d, 4/d, \ldots$. The factor of 2 is essential as each $i$ corresponds to a pair of dimensions.</span>

<span style="font-size: 14px;">**Confusing the table with the full rotation.** The precomputed tables store $\cos(p \cdot \theta_i)$ and $\sin(p \cdot \theta_i)$, but applying RoPE also requires rearranging the input: $q'_{2i} = q_{2i}\cos\alpha - q_{2i+1}\sin\alpha$ and $q'_{2i+1} = q_{2i}\sin\alpha + q_{2i+1}\cos\alpha$. The table is the precomputation step; the rotation application is a separate operation that consumes it.</span>

---
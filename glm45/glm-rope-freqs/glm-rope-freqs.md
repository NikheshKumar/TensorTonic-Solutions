# <span style="font-size: 20px;">RoPE Frequency Table</span>

<span style="font-size: 14px;">Rotary Position Embedding (RoPE) injects absolute position into a Transformer by rotating pairs of query and key dimensions by position-dependent angles. The cosine and sine tables built here are precomputed once and reused at every attention layer of GLM-4.5, which keeps the runtime cost of positional encoding negligible.</span>

---

## <span style="font-size: 16px;">What RoPE Solves</span>

<span style="font-size: 14px;">Self-attention is permutation invariant. Without positional information the model treats "the cat sat on the mat" and "mat the on sat cat the" as the same multiset of tokens. Some signal must be added so attention scores depend on token order.</span>

<span style="font-size: 14px;">RoPE (Su et al., 2021) provides that signal through rotation rather than addition. Each query and key vector is reshaped into pairs of dimensions, and each pair is rotated by an angle that depends on the token position. The dot product between a query at position $p$ and a key at position $q$ then naturally encodes the relative offset $p - q$, without the model ever storing absolute positions inside attention scores.</span>

* <span style="font-size: 14px;">**Relative position for free:** $q_p^\top k_q$ only depends on $p - q$, not on $p$ and $q$ separately.</span>
* <span style="font-size: 14px;">**No extra parameters:** the cos/sin tables are deterministic functions of position and dimension index.</span>
* <span style="font-size: 14px;">**Streaming friendly:** new positions can be appended at inference without rebuilding the embedding table.</span>

---

## <span style="font-size: 16px;">The Frequency Formula</span>

<span style="font-size: 14px;">For a head dimension $d$ (an even integer) we form $d/2$ dimension pairs, indexed by $i = 0, 1, \ldots, d/2 - 1$. Each pair gets its own inverse frequency:</span>

$$
\theta_i = \frac{1}{\text{rope\_theta}^{\,2i / d}}
$$

<span style="font-size: 14px;">The rotation angle for position $p$ and pair $i$ is the product:</span>

$$
\text{angle}(p, i) = p \cdot \theta_i
$$

<span style="font-size: 14px;">The two output tables are then:</span>

$$
\text{cos}[p, i] = \cos(p \cdot \theta_i), \qquad \text{sin}[p, i] = \sin(p \cdot \theta_i)
$$

<span style="font-size: 14px;">Both have shape $(\texttt{n\_tokens}, d / 2)$. The naming `rope_theta` matches the GLM-4.5 config field; the original RoPE paper called it `base`.</span>

---

## <span style="font-size: 16px;">Geometric Interpretation</span>

<span style="font-size: 14px;">View each pair $(x_{2i}, x_{2i+1})$ of a query or key vector as a point in the 2D plane. RoPE rotates this 2D point by the angle $p \cdot \theta_i$:</span>

$$
\begin{pmatrix} x'_{2i} \\ x'_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos(p\theta_i) & -\sin(p\theta_i) \\ \sin(p\theta_i) & \cos(p\theta_i) \end{pmatrix} \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}
$$

<span style="font-size: 14px;">A rotation matrix preserves the norm, so RoPE never changes the magnitude of a query or key, only its direction. The model's existing magnitudes (which encode content) are untouched, and the rotation overlays a position signal on top.</span>

<span style="font-size: 14px;">Now consider the dot product of two rotated vectors at positions $p$ and $q$. Because rotation matrices compose, the inner product of $R(p\theta_i) v_1$ and $R(q\theta_i) v_2$ equals $v_1^\top R((q - p)\theta_i) v_2$. Attention scores therefore depend on $q - p$, the relative offset. This is the central elegance of RoPE.</span>

---

## <span style="font-size: 16px;">Why a Geometric Frequency Schedule</span>

<span style="font-size: 14px;">The factor $\text{rope\_theta}^{2i/d}$ in the denominator spreads the inverse frequencies over many orders of magnitude. For the standard $\text{rope\_theta} = 10000$ and $d = 128$, $\theta_0 = 1$ and $\theta_{63} \approx 10^{-4}$.</span>

* <span style="font-size: 14px;">**High-frequency pairs** (small $i$) rotate fast: at $p = 1$ they already swing by about one radian, so they distinguish adjacent positions clearly.</span>
* <span style="font-size: 14px;">**Low-frequency pairs** (large $i$) rotate slowly: even at $p = 1000$ they have barely turned, so they distinguish far-apart positions without aliasing.</span>
* <span style="font-size: 14px;">**Multi-scale code:** together the pairs form a positional fingerprint analogous to a base-$\text{rope\_theta}$ number. Each pair is one "digit" at a different scale.</span>

<span style="font-size: 14px;">The schedule mirrors the sinusoidal positional encoding of Vaswani et al. (2017). The two differ in how the signal enters the model: sinusoidal encodings are added to embeddings before the first layer, RoPE rotates queries and keys at every layer. RoPE has emerged as the dominant choice in modern LLMs (LLaMA, Mistral, Qwen, GLM, DeepSeek) because the rotation gives clean relative-position behavior and survives layer normalization.</span>

---

## <span style="font-size: 16px;">Role in GLM-4.5</span>

<span style="font-size: 14px;">GLM-4.5 uses RoPE as its positional encoding, applied inside multi-head attention before the QK dot product. Two GLM-4.5 specific details are worth noting:</span>

* <span style="font-size: 14px;">**Partial RoPE.** GLM-4.5 applies RoPE only to a leading slice of each head, not the full $d$ dimensions. The cos/sin tables built here have width $d/2$ over the full head; the slicing is handled later when the rotation is applied. This is identical to the cos/sin computation in vanilla RoPE.</span>
* <span style="font-size: 14px;">**No YaRN or NTK scaling.** The vanilla base-frequency formula above is what GLM-4.5 ships. There is no temperature, no piecewise interpolation, no extrapolation scaling. The table is exactly the schedule from the original RoPE paper, with whatever `rope_theta` the config sets.</span>

<span style="font-size: 14px;">The table only needs to be built once per model load. It is cached on the device and broadcast across the batch and head dimensions when the rotation is applied. Building it as float32 (or even float64) costs a few kilobytes and a few microseconds, even for context lengths of tens of thousands.</span>

---

## <span style="font-size: 16px;">Step-by-Step Construction</span>

<span style="font-size: 14px;">1. **Build the index vector** $i = (0, 1, \ldots, d/2 - 1)$ as a 1D tensor of length $d/2$.</span>

<span style="font-size: 14px;">2. **Compute inverse frequencies** $\theta_i = 1 / \text{rope\_theta}^{2i/d}$. This is a 1D tensor of length $d/2$. The exponent $2i/d$ is a float division: do not let integer division collapse it.</span>

<span style="font-size: 14px;">3. **Build the position vector** $p = (0, 1, \ldots, \texttt{n\_tokens} - 1)$ as a 1D tensor.</span>

<span style="font-size: 14px;">4. **Outer product:** multiply positions (shape $(\texttt{n\_tokens}, 1)$) with inverse frequencies (shape $(1, d/2)$). The broadcast result has shape $(\texttt{n\_tokens}, d/2)$. Equivalent forms: `torch.outer(p, theta)` or `p.unsqueeze(1) * theta.unsqueeze(0)`.</span>

<span style="font-size: 14px;">5. **Apply cos and sin element-wise** to the angle matrix and return them as a tuple `(cos, sin)`.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Take $\texttt{n\_tokens} = 4$, $d = 4$, $\text{rope\_theta} = 10000$. There are $d/2 = 2$ pairs.</span>

* <span style="font-size: 14px;">$\theta_0 = 1 / 10000^{0/4} = 1.0$</span>
* <span style="font-size: 14px;">$\theta_1 = 1 / 10000^{2/4} = 1 / 100 = 0.01$</span>

<span style="font-size: 14px;">Position vector $p = (0, 1, 2, 3)$. Angle matrix is the outer product:</span>

$$
\text{angles} = \begin{pmatrix} 0 & 0 \\ 1 & 0.01 \\ 2 & 0.02 \\ 3 & 0.03 \end{pmatrix}
$$

<span style="font-size: 14px;">Cosine table (rounded):</span>

$$
\cos(\text{angles}) \approx \begin{pmatrix} 1.0000 & 1.0000 \\ 0.5403 & 1.0000 \\ -0.4161 & 0.9998 \\ -0.9900 & 0.9996 \end{pmatrix}
$$

<span style="font-size: 14px;">Sine table (rounded):</span>

$$
\sin(\text{angles}) \approx \begin{pmatrix} 0.0000 & 0.0000 \\ 0.8415 & 0.0100 \\ 0.9093 & 0.0200 \\ 0.1411 & 0.0300 \end{pmatrix}
$$

<span style="font-size: 14px;">Notice how the first column changes fast (the high-frequency pair) while the second column barely moves: 0.01, 0.02, 0.03. That second pair is the slow clock that will still be useful at much larger positions.</span>

---

## <span style="font-size: 16px;">Comparison with Sinusoidal Encodings</span>

<span style="font-size: 14px;">The original Transformer used additive sinusoidal positional encodings with the same kind of geometric frequency schedule. There the encoding $\text{PE}(p, 2i) = \sin(p / 10000^{2i/d})$ and $\text{PE}(p, 2i + 1) = \cos(p / 10000^{2i/d})$ is added to the token embedding before layer 1.</span>

* <span style="font-size: 14px;">**Sinusoidal:** position lives in the residual stream, gets mixed into queries, keys and values, and competes with content for representational capacity.</span>
* <span style="font-size: 14px;">**RoPE:** position only affects queries and keys, and only by rotating them. Values are untouched. The signal is reinjected at every layer because the rotation is part of attention, not the input embedding.</span>

<span style="font-size: 14px;">Empirically, RoPE generalizes more cleanly to longer sequences and pairs better with extrapolation tricks like NTK scaling and YaRN, even though GLM-4.5 itself uses the vanilla form.</span>

---

## <span style="font-size: 16px;">Effect of `rope_theta`</span>

<span style="font-size: 14px;">The base frequency controls how fast inverse frequencies shrink across pairs. A larger `rope_theta` produces smaller inverse frequencies everywhere, which means slower rotation, which means a longer effective context before pairs alias.</span>

* <span style="font-size: 14px;">Original RoPE / LLaMA 1: $\text{rope\_theta} = 10000$, paired with $\sim$2K context.</span>
* <span style="font-size: 14px;">LLaMA 3 and modern long-context models: $\text{rope\_theta} = 500000$ to push the effective window past 100K tokens.</span>
* <span style="font-size: 14px;">GLM-4.5: uses a large base in the config to support its long context, with the same simple formula.</span>

<span style="font-size: 14px;">Choosing the right base is essentially picking how many distinct positions the slowest pair can represent before it wraps around.</span>

---

## <span style="font-size: 16px;">Where the Table Plugs Into Attention</span>

<span style="font-size: 14px;">Once the cos and sin tables exist, applying RoPE to a query tensor $Q$ of shape $(\texttt{batch}, \texttt{n\_heads}, \texttt{n\_tokens}, d)$ proceeds as follows. The same recipe applies to $K$.</span>

* <span style="font-size: 14px;">**Split into pair halves.** Reshape the last axis so pair $i$ contributes one even component and one odd component. Two common conventions exist: the "interleaved" form $(x_0, x_1, x_2, x_3, \ldots)$ where pair $i$ is $(x_{2i}, x_{2i+1})$, and the "split-half" form $(x_0, \ldots, x_{d/2 - 1}, x_{d/2}, \ldots)$ where pair $i$ is $(x_i, x_{i + d/2})$. GLM-4.5 follows the split-half convention used by LLaMA.</span>
* <span style="font-size: 14px;">**Broadcast the tables.** The cos and sin tables have shape $(\texttt{n\_tokens}, d/2)$. Unsqueeze a batch and head dimension and either repeat or broadcast along them.</span>
* <span style="font-size: 14px;">**Apply the 2D rotation.** For each pair the new components are $x' = x \cos - y \sin$ and $y' = x \sin + y \cos$. Implemented as a single elementwise expression on the reshaped tensor.</span>

<span style="font-size: 14px;">The table itself is independent of these implementation choices: cos and sin still depend only on position and the inverse frequency schedule. That separation of concerns is why precomputing the table is clean.</span>

---

## <span style="font-size: 16px;">Complexity and Memory</span>

* <span style="font-size: 14px;">**Build time:** $O(\texttt{n\_tokens} \cdot d / 2)$ floating point operations, dominated by the outer product. For typical values like $\texttt{n\_tokens} = 8192$ and $d = 128$, the table holds about half a million floats per side, or roughly 4 MB total in float32. Negligible compared to the model weights.</span>
* <span style="font-size: 14px;">**Apply time:** per token, per head, $O(d)$ work for the rotation. The cost is dominated by the QK projection and the attention matmul, not by RoPE.</span>
* <span style="font-size: 14px;">**Storage:** keep one copy on the GPU. There is no need to checkpoint or recompute it during the backward pass because the table is a constant function of position.</span>

<span style="font-size: 14px;">If the model can stream past the precomputed length, either rebuild the table for a longer max length or compute the rotation on the fly for new positions. The formula is cheap enough that both approaches are fine.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Wrong exponent.** Using $\text{rope\_theta}^{i/d}$ instead of $\text{rope\_theta}^{2i/d}$ halves the frequency range and produces wrong rotations. The factor of 2 is essential because there are $d/2$ pairs but $d$ underlying dimensions.</span>
* <span style="font-size: 14px;">**Swapping cos and sin.** The two tables are not interchangeable: cos sits on the diagonal of the 2D rotation matrix, sin on the off-diagonal. Returning them in the wrong order silently produces a different (and incorrect) rotation.</span>
* <span style="font-size: 14px;">**Using `arange(0, head_dim)` instead of `arange(0, head_dim, 2)`.** The frequency tensor must have length $d/2$, one entry per pair, not $d$. Indexing every dimension instead of every pair doubles the output width and breaks shape contracts downstream.</span>
* <span style="font-size: 14px;">**Forgetting the outer product.** The angle tensor is 2D, not 1D. A common slip is to skip the broadcast and return $\cos(\theta)$ and $\sin(\theta)$ of shape $(d/2,)$, which has no position dimension at all.</span>
* <span style="font-size: 14px;">**Multiplying angles by $2\pi$.** RoPE angles are already in radians; positions are integers, inverse frequencies have units of radians-per-token. Adding a $2\pi$ factor would scale every rotation by roughly 6.28 and ruin the schedule.</span>
* <span style="font-size: 14px;">**Hardcoding `rope_theta`.** It is a configurable knob, not a magic constant. GLM-4.5, LLaMA 3, and others ship different values. Always read it from the model config (or argument), never bake in 10000.</span>
* <span style="font-size: 14px;">**Integer division of the exponent.** Writing `(2 * i) // head_dim` collapses the exponent to 0 or 1, producing only two distinct frequencies. Keep the division in floating point.</span>
* <span style="font-size: 14px;">**Float32 precision at long context.** At positions beyond about $2^{24}$ the float32 representation of $p$ starts losing the unit step. For very long contexts compute the tables in float64 and cast down only at the end. The reference here uses float64 for that reason.</span>

---

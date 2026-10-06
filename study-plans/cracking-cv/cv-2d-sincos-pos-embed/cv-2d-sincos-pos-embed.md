# <span style="font-size: 20px;">2D Sinusoidal Positional Embedding</span>

<span style="font-size: 14px;">2D sin-cos positional embedding is a fixed (non-learned) encoding that tells a Vision Transformer where each patch token sits on the image grid. It extends the 1D sinusoidal scheme of the original Transformer (Vaswani et al., 2017) to two spatial axes and is the default position encoding in ViT-MAE (He et al., 2021), DINOv2, and many MAE-style models, chosen because it requires no parameters and generalizes across image resolutions.</span>

---

## <span style="font-size: 16px;">Why Positions Are Needed</span>

<span style="font-size: 14px;">Self-attention is permutation-equivariant: it has no built-in notion of order, so shuffling the input tokens shuffles the output identically. After patchify turns an image into a flat sequence of patch tokens, all spatial structure is lost. Without a position signal the model cannot tell a top-left patch from a bottom-right one. Positional embeddings restore that information by adding a unique, location-dependent vector to each token.</span>

<span style="font-size: 14px;">ViT and BERT use **learned** positional embeddings, one trainable vector per position. Sinusoidal embeddings instead use a **fixed** deterministic function of position. The MAE paper adopts 2D sin-cos embeddings specifically so that no positional parameters are trained, which simplifies the masked-autoencoder setup and makes the encoder behave consistently regardless of how many patches are visible.</span>

<span style="font-size: 14px;">It is worth stressing that without any position signal, an attention layer treats its input as an unordered set: feeding the patches in any permutation produces the same set of outputs, just permuted the same way. For text this would lose word order; for images it loses the entire 2D layout. The position embedding is the single mechanism that re-injects geometry, and the sin-cos variant does so with a hand-designed function rather than learned parameters.</span>

---

## <span style="font-size: 16px;">The 1D Building Block</span>

<span style="font-size: 14px;">The 2D scheme is built from a 1D encoder applied to each axis. For a single axis with half-dimension $d = D / 2$, define a geometric set of frequencies:</span>

$$
\omega_i = \frac{1}{10000^{2 i / d}}, \quad i \in \{0, 1, \ldots, d/2 - 1\}
$$

<span style="font-size: 14px;">For a position $p \in \{0, 1, \ldots, L-1\}$ along an axis of length $L$, the 1D feature concatenates all the sines followed by all the cosines:</span>

$$
\bigl[ \sin(p\,\omega_0), \ldots, \sin(p\,\omega_{d/2-1}),\ \cos(p\,\omega_0), \ldots, \cos(p\,\omega_{d/2-1}) \bigr]
$$

<span style="font-size: 14px;">Each frequency $\omega_i$ produces one sine and one cosine, so the $d/2$ frequencies yield exactly $d$ output values. The frequencies span a wide geometric range: $\omega_0 = 1$ gives a slowly varying signal (low frequency, long wavelength) while the largest $i$ gives a rapidly oscillating signal (high frequency, short wavelength). The wavelengths form a geometric progression from $2\pi$ up to $10000 \cdot 2\pi$.</span>

<span style="font-size: 14px;">The multi-frequency design is what makes the code informative. A single frequency would alias: positions $p$ and $p + 2\pi/\omega$ would map to the same value. By stacking many frequencies, the combined vector is unique for every integer position within the representable range, much like how a clock's hour, minute, and second hands together pin down a unique time even though each hand alone is periodic. Low frequencies disambiguate coarse location (which region of the image) and high frequencies disambiguate fine location (which exact patch), so a downstream attention head can read off position at whatever granularity it needs.</span>

<span style="font-size: 14px;">The base constant $10000$ is inherited directly from Vaswani et al., 2017. It sets how slowly the lowest frequency varies and therefore the longest wavelength the encoding can represent unambiguously. For typical ViT grids of a few dozen patches per side, $10000$ is far larger than needed, so the slowest channels barely change across the grid and act as near-constant offsets, while the faster channels carry the discriminative signal.</span>

---

## <span style="font-size: 16px;">Combining Two Axes</span>

<span style="font-size: 14px;">For a 2D grid of size $H \times W$ and embedding dim $D$ divisible by 4, split $D$ in half. The first $D/2$ channels encode the **row** coordinate and the second $D/2$ encode the **column** coordinate. Each half is computed by the 1D building block above with $d = D/2$, so within each half there are $d/2 = D/4$ frequencies, which is why $D$ must be divisible by 4.</span>

<span style="font-size: 14px;">For each grid cell at row $r$ and column $c$, the full embedding is the concatenation:</span>

$$
\text{emb}(r, c) = \bigl[\, \text{emb}_{1D}(r) \;\|\; \text{emb}_{1D}(c) \,\bigr] \in \mathbb{R}^{D}
$$

<span style="font-size: 14px;">The grid is walked in row-major order (row slow, column fast), producing a sequence of $H \cdot W$ vectors of length $D$. This row-major flattening must match the patch ordering from patchify exactly, so that token $k$ in the sequence receives the position vector for the patch at the same location.</span>

<span style="font-size: 14px;">The separable, concatenated design is deliberate. Because the row code occupies its own $D/2$ channels and the column code occupies the rest, the two axes never interfere: an attention head can attend to the row channels alone to compare vertical position, or the column channels alone for horizontal position. An alternative would be to multiply or sum the per-axis codes into a shared block, but that entangles the axes and makes single-axis reasoning harder. Concatenation keeps the two coordinates linearly separable, which is why ViT-MAE and DINOv2 use this exact layout.</span>

<span style="font-size: 14px;">When a `[CLS]` token is present, its position is usually a separate fixed or zero vector prepended to the $H \cdot W$ grid codes, since it has no spatial location. MAE's encoder, which has no class token during pretraining, simply uses the $H \cdot W$ grid codes directly.</span>

---

## <span style="font-size: 16px;">Why It Generalizes Across Resolutions</span>

<span style="font-size: 14px;">The frequencies $\omega_i$ depend only on the embedding dimension, not on $H$ or $W$. The encoding is a pure function of the integer coordinate, so it is defined for any position, including ones never seen at training time. If a model is pretrained on a $14 \times 14$ patch grid and then run on a larger image producing a $20 \times 20$ grid, the sin-cos formula simply evaluates at the new coordinates with no interpolation and no retraining.</span>

<span style="font-size: 14px;">Learned embeddings cannot do this directly: there is no trained vector for position 15 if training only ever reached position 13, so practitioners must interpolate the learned table, which is approximate. Fixed sin-cos embeddings sidestep the problem entirely, which is the practical reason MAE and DINOv2 prefer them when variable resolution or test-time scaling matters.</span>

<span style="font-size: 14px;">A second classical property, inherited from the 1D Vaswani encoding: for any fixed offset $\Delta$, the embedding of position $p + \Delta$ is a linear function of the embedding of $p$, because shifting the argument of sine and cosine is a rotation. Concretely, $\sin(\omega(p+\Delta)) = \sin(\omega p)\cos(\omega\Delta) + \cos(\omega p)\sin(\omega\Delta)$, a fixed $2 \times 2$ rotation acting on the $(\sin, \cos)$ pair for each frequency. This lets attention learn to attend by relative offset, since a relative shift corresponds to a fixed linear map applied uniformly across positions, independent of the absolute location $p$.</span>

<span style="font-size: 14px;">This relative-shift property is the conceptual ancestor of rotary position embeddings (RoPE) now common in language models: RoPE applies the same rotation idea directly inside the query and key vectors rather than adding a position code to the input. Understanding the sin-cos rotation here makes the connection to those modern variants explicit.</span>

---

## <span style="font-size: 16px;">How the Embedding Is Used</span>

<span style="font-size: 14px;">The position embedding is **added**, not concatenated, to the patch embeddings: $z_k = \text{patch\_embed}_k + \text{pos}_k$. Both live in $\mathbb{R}^D$, so addition keeps the token width fixed. Because the sin-cos values are fixed, they are typically precomputed once and stored as a buffer rather than a parameter.</span>

<span style="font-size: 14px;">In MAE the embedding is added to all tokens before masking, so even after 75 percent of patches are dropped the remaining tokens still carry correct absolute positions. The decoder reinserts mask tokens at their grid locations and again adds the same fixed sin-cos table, letting it know where each missing patch belongs. This is precisely why a parameter-free, position-pure code is convenient for MAE: the same table serves both the partial-input encoder and the full-grid decoder without any learned position parameters to keep in sync.</span>

<span style="font-size: 14px;">Adding rather than concatenating the position code is itself a design choice. Addition keeps the model width at $D$ and lets the network blend content and position in whatever proportion it learns, since both share the same channels. Concatenation would grow the width and force a hard separation. The original Transformer chose addition and ViT-MAE follows suit; in practice the patch-embedding projection learns to leave headroom for the additive position signal.</span>

---

## <span style="font-size: 16px;">Step by Step</span>

<span style="font-size: 14px;">1. **Set the half-dim**: $d = D / 2$, with $d/2 = D/4$ frequencies per axis. Require $D \bmod 4 = 0$.</span>

<span style="font-size: 14px;">2. **Build frequencies**: $\omega_i = 10000^{-2i/d}$ for $i = 0, \ldots, d/2 - 1$.</span>

<span style="font-size: 14px;">3. **Encode the row axis**: for each row index $r \in \{0, \ldots, H-1\}$ form $[\sin(r\omega_0), \ldots, \cos(r\omega_{d/2-1})]$, giving an $(H, d)$ table.</span>

<span style="font-size: 14px;">4. **Encode the column axis**: identically for $c \in \{0, \ldots, W-1\}$, giving a $(W, d)$ table.</span>

<span style="font-size: 14px;">5. **Walk the grid row-major**: for each cell $(r, c)$ concatenate the row code and the column code into a length-$D$ vector.</span>

<span style="font-size: 14px;">6. **Stack**: collect the $H \cdot W$ vectors in order into the final $(H \cdot W, D)$ array, ready to add to patch embeddings.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Take $D = 4$, so each half has $d = D/2 = 2$ and each half uses $d/2 = 1$ frequency, namely $\omega_0 = 1/10000^{0} = 1$. The 1D encoder for a position $p$ is $[\sin(p), \cos(p)]$.</span>

<span style="font-size: 14px;">Consider a $2 \times 2$ grid and walk it row-major as cells $(0,0), (0,1), (1,0), (1,1)$:</span>

* <span style="font-size: 14px;">**Cell (0,0):** row $r=0$ gives $[\sin 0, \cos 0] = [0, 1]$, col $c=0$ gives $[0, 1]$, concatenated to $[0, 1, 0, 1]$</span>
* <span style="font-size: 14px;">**Cell (0,1):** row $r=0$ gives $[0, 1]$, col $c=1$ gives $[\sin 1, \cos 1] \approx [0.8415, 0.5403]$, so $[0, 1, 0.8415, 0.5403]$</span>
* <span style="font-size: 14px;">**Cell (1,0):** row $r=1$ gives $[0.8415, 0.5403]$, col $c=0$ gives $[0, 1]$, so $[0.8415, 0.5403, 0, 1]$</span>
* <span style="font-size: 14px;">**Cell (1,1):** both axes at 1 give $[0.8415, 0.5403, 0.8415, 0.5403]$</span>

<span style="font-size: 14px;">The output is a $(4, 4)$ array. Note that the row part of cell $(0,0)$ and $(0,1)$ is identical because they share a row, while their column parts differ; this is exactly the separable structure that lets attention reason about the two axes independently. Likewise cells $(0,0)$ and $(1,0)$ share an identical column part and differ only in the row part. With a realistic $D = 768$ there are $192$ frequencies per axis and the codes become densely unique, but the $2 \times 2$ case shows the mechanics clearly with values a reader can verify by hand.</span>

---

## <span style="font-size: 16px;">Comparison With Alternatives</span>

* <span style="font-size: 14px;">**Learned absolute (ViT, BERT).** One trainable vector per position. Slightly more expressive on the training grid but needs interpolation to change resolution and adds parameters.</span>
* <span style="font-size: 14px;">**1D sin-cos flattened.** Treat the patch sequence as a single 1D line of length $H \cdot W$. Simpler but throws away the 2D adjacency: patches in the same column on different rows get unrelated codes.</span>
* <span style="font-size: 14px;">**2D sin-cos (this method).** Separable per-axis encoding preserves the grid structure while staying parameter-free and resolution-agnostic. The standard MAE choice.</span>
* <span style="font-size: 14px;">**Relative position bias (Swin).** Adds a learned bias to attention scores based on the offset between query and key patches, rather than to the token embeddings. Often combined with or used instead of absolute embeddings.</span>
* <span style="font-size: 14px;">**Rotary embeddings (RoPE).** Rotates query and key vectors by an angle proportional to position so the dot product depends only on relative offset. Increasingly ported into vision Transformers for its length generalization, and conceptually a descendant of the sin-cos rotation property.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Wrong sin/cos interleaving.** This convention puts all sines first, then all cosines, within each axis half. Some implementations interleave them as $[\sin\omega_0, \cos\omega_0, \sin\omega_1, \ldots]$. Both are valid but they are not interchangeable, and mismatching the convention against a pretrained checkpoint silently corrupts every position.</span>
* <span style="font-size: 14px;">**Forgetting $D$ divisible by 4.** Each axis takes $D/2$ channels, and each axis needs an even count for its sine/cosine pair, so $D/2$ must be even, meaning $D$ must be a multiple of 4. Using a $D$ that is only divisible by 2 produces an off-by-one frequency count and a shape mismatch.</span>
* <span style="font-size: 14px;">**Mismatched flatten order.** The position grid must be flattened in the same row-major order as the patches. If patchify is row-major but the position table is built column-major, every token gets the wrong position and the model trains on scrambled geometry.</span>
* <span style="font-size: 14px;">**Treating it as learnable.** The embedding is a fixed buffer, not a parameter. Accidentally registering it as a trainable tensor lets gradients drift the carefully chosen frequencies away from their geometric spacing, destroying the relative-offset property and the resolution generalization that motivated the scheme.</span>

---
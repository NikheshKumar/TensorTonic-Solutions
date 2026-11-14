# Per-Layer Embeddings

Per-layer embeddings give every transformer layer in EmbeddingGemma 2 its own small input that is computed straight from the input token embeddings. The block adds a gated, normalized update to the residual stream after each layer. This problem covers one layer of that block in the projection-only form that the paper introduces, where the per-layer signal comes entirely from the shared input embeddings $T_{\text{in}}$ of shape $(B, L, d_M)$.

## A direct line from the input to every layer

In a plain transformer, the only path from the input to layer $l$ runs through the residual stream. Every earlier layer writes into that stream, so token identity gets harder to read as depth grows. Gemma 3n introduced per-layer embeddings to fix this. Each layer receives a fresh, layer-specific view of the input, so it can recover token-level detail without depending on what earlier layers kept.

The view lives in a per-layer width $d_P$. The projection $p_l$ maps each $d_M$-dimensional input embedding to $d_P$ channels. The gate $g_l$ reads the layer's own output $H'_l$, so the layer decides which of those channels it wants right now. The map $u_l$ brings the gated signal back to $d_M$ so it can be added to the stream.

## Why the lookup table could go

Gemma 3n and Gemma 4 build the per-layer input from two parts. One part is a learned projection of the main token embedding, and the other is a large lookup table used only for per-layer embeddings. That table has rows for text tokens alone. An image patch or an audio frame enters the backbone as a soft token produced by an encoder, so it has no row to look up, and the table contributes nothing for it.

EmbeddingGemma 2 removes the table and keeps only the projection. Because the vision and audio soft tokens are already projected into $T_{\text{in}}$, every modality now drives the per-layer path in the same way. The ablation in Schechter Vera et al., 2026 shows that the simpler block is also better. The projection-only model scores 55.46 overall on MMEB-v2 against 55.07 for the version with the lookup table, with the largest gains on video (+0.69) and visual documents (+0.61). This change fits the paper's broader aim of one compact backbone that treats every modality alike and can be loaded without its encoders for text.

## Two norms and one gate

Each RMSNorm fixes the size of a vector and leaves its direction to be learned. The first norm makes $P_l$ independent of how large the projected input happens to be, so a token with a very large embedding cannot dominate the update. The second norm fixes the size of $U_l$ for every token, which keeps the residual stream stable across all 24 layers.

The second norm has a consequence worth noticing. Since $u_l$ is linear, multiplying $G_l \odot P_l$ by a positive number for one token leaves $U_l$ unchanged. What the gate really controls is the mix of channels, so it acts as a selector more than a volume knob. Without the gate, every layer would receive the same mix of per-layer channels whatever it was currently computing.

The paper names the operations but leaves several details open, so the exercise fixes them. The exercise chooses bias-free linear maps, the exact erf form of GELU, and an RMSNorm that multiplies by the scale vector itself rather than by one plus the scale. The paper uses $d_P = d_M = 512$, and the exercise lets the two widths differ so that every matrix shape is unambiguous.

## A gate that flips a channel

Take one token with $d_M = d_P = 2$, identity weights, unit scales and $\epsilon = 10^{-6}$. Let the input embedding be $(1, 3)$ and the layer output be $(1, -1)$. The projected input has mean square $(1 + 9)/2 = 5$, so $P_l = (0.4472, 1.3416)$. The gate gives $\operatorname{GELU}(1) = 0.8413$ and $\operatorname{GELU}(-1) = -0.1587$.

The product is $G_l \odot P_l = (0.3763, -0.2129)$. The second channel carried the larger input value, yet the gate shrank it and flipped its sign. The mean square of the product is $0.0935$, so the second norm rescales it to $U_l = (1.2309, -0.6963)$. Adding the layer output gives $H_l = (2.2309, -1.6963)$.

## Common mistakes to avoid

- **Projecting the layer output into the per-layer space.** The per-layer input must come from $T_{\text{in}}$, which is the whole point of giving each layer a direct line to the tokens.
- **Gating with the input embeddings.** The gate reads $H'_l$, so the same input produces different updates at different depths.
- **Swapping GELU for sigmoid or SiLU.** A sigmoid never reaches zero, so a zero layer output would still let the per-layer signal through.
- **Using the tanh approximation of GELU.** The exercise uses the exact erf form, and the two differ by enough to change the result.
- **Normalizing before a projection.** Each RMSNorm acts on the output of $p_l$ or $u_l$, and moving it in front of the map changes the size of the update.
- **Using one plus the scale.** The exercise multiplies by $s_P$ and $s_U$ directly, so adding one to the scale inflates every channel.
- **Returning the update alone.** The block is a residual branch, so $H'_l$ must be added back.
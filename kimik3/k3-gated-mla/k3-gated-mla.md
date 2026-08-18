# Gated MLA

Gated Multi-head Latent Attention is ordinary global attention with a compact storage path and a learned output gate. Each token is compressed into one latent vector. Keys and values are reconstructed from that latent representation when attention runs, so the model does not need separate full-width cached content for every head.

## Start from familiar attention

Attention still follows the usual story. A query describes what a token wants, keys describe what can be matched, and values contain the information returned. For each head, scaled dot-product attention computes

$$
A = \operatorname{softmax}\left(\frac{QK^{\mathsf T}}{\sqrt{D_h}} + M\right)
$$

and the head context is $AV$. The mask $M$ blocks later positions in causal mode. Softmax is taken over key positions so every query gets a distribution over the tokens it may read.

What changes is how keys and values are produced. Queries come directly from the hidden states, while each token first creates a latent representation

$$
C = XW_c^{\mathsf T}
$$

The latent vector is then expanded in two different ways:

$$
K=CW_k^{\mathsf T}, \qquad V=CW_v^{\mathsf T}
$$

One compressed vector therefore carries the source information needed to reconstruct both keys and values. The function returns this latent sequence as the cache.

## Why the latent representation helps

Without compression, storing keys and values means retaining a larger set of features for every earlier token. MLA stores the smaller latent vector instead. The key and value up-projections can recover the representations needed by attention.

For this problem, do not confuse compression with averaging. Every token keeps its own latent vector, and the sequence length remains unchanged. Only the feature width is reduced.

Kimi K3 periodically uses Gated MLA for unrestricted global content interaction. These layers use no explicit positional encoding. The exercise therefore does not add rotary embeddings, learned position vectors, or any other position transformation. Causal order is enforced only by the mask when causal mode is enabled.

## Split attention into heads

The projected query, key, and value tensors each have model width. Divide that width evenly across the requested number of heads. Each head performs its own attention calculation using feature width $D_h$.

The heads must remain independent until their context vectors have been computed. Afterward, place their features back beside each token to recover model width. This is a reshape and transpose operation, not a sum across heads.

## The channel-wise output gate

The ungated attention context is multiplied by a gate derived from the original hidden state:

$$
Y = \left[\operatorname{sigmoid}(XW_g^{\mathsf T})\odot\widetilde{O}\right]W_o^{\mathsf T}
$$

Sigmoid gives one number between zero and one for each token channel. A channel with a gate near zero is suppressed, while a gate near one passes through almost unchanged. Because the gate has full model width, different channels of the same token can be controlled differently.

Apply this gate before the final output projection. The projection then mixes the gated channels into the final representation.

## A small causal example

Consider three tokens. In causal mode, the first query may attend only to token 1. The second may attend to tokens 1 and 2. The third may attend to all three. Each head produces a lower-triangular attention pattern because later key positions are blocked before softmax.

Suppose one channel of the joined attention context for token 2 is $1.6$, and its gate logit is zero. Sigmoid of zero is $0.5$, so that channel becomes $0.8$ before the output projection. The gate changes what is passed onward without changing the attention probabilities themselves.

In non-causal mode, all three queries may use all three keys. The latent cache is identical in both cases because the cache depends on the hidden states and latent down-projection, not on the mask.

## Implementation order

- Project the hidden states into queries and latent vectors.
- Expand the latent vectors into keys and values.
- Split query, key, and value features into heads.
- Compute scaled scores, apply a causal mask only when requested, and normalize over key positions.
- Combine values, join the head outputs, and restore model width.
- Compute the channel gate from the original hidden states, multiply it with the joined context, and apply the output projection.
- Return the output followed by the latent cache.

## Common mistakes to avoid

- **Caching reconstructed keys and values.** This task returns the compact latent tensor $C$.
- **Adding positional encoding.** The problem specifically implements the NoPE form used here.
- **Masking after softmax.** Blocked positions must be removed before normalization.
- **Applying one gate per token.** The gate is channel-wise and has model width.
- **Gating after the output projection.** Follow the stated order: context, gate, then output projection.

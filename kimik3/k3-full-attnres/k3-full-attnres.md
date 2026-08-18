# Full Attention Residuals

A normal residual stream carries one running representation through the depth of a network. Full Attention Residuals keeps the embedding and every earlier layer output available, then lets the current layer learn how much to retrieve from each depth. It is attention across layers rather than attention across token positions.

## The depth sources

The first source is always the token embedding. Every preceding layer output adds another source. If three layers have already run, the current layer can retrieve from four places: the original embedding and the three layer outputs.

Each source is used in two roles. Its normalized form helps determine an attention score, while its original unnormalized form supplies the value that will be mixed into the result.

This distinction is central to the exercise. Normalizing the keys prevents a large-magnitude layer output from winning only because its numbers are bigger. Keeping the values raw preserves the actual representation produced by that layer.

## One learned query for the current layer

The pseudo-query is a learned vector associated with the layer doing the retrieval. The same pseudo-query is compared with every depth source at every token position. It does not come from the token content.

For source $i$, compute

$$
\ell_i = w_l^{\mathsf T}\operatorname{RMSNorm}(k_i)
$$

Softmax across the source axis turns these logits into depth weights:

$$
\alpha_i = \frac{\exp(\ell_i)}{\sum_j \exp(\ell_j)}
$$

The retrieved representation is the weighted mixture

$$
h_l = \sum_i \alpha_i v_i
$$

For every batch item and token, the weights sum to one. Different token positions can still receive different depth weights because the source representations being scored are different.

## Why this is not a standard residual sum

An ordinary residual connection adds a new transformation to the current stream with a fixed coefficient of one. Over many layers, all earlier information is compressed into the latest representation.

Full Attention Residuals exposes the earlier representations directly. A layer can emphasize the embedding for one token and a recent layer output for another token. Softmax also keeps the scale of the mixture controlled because it forms a convex combination of the sources.

The operation does not attend across sequence positions. Source 2 at token position 5 is compared only as a depth source for token position 5. Sequence mixing has already happened inside the model layers that produced these values.

## A three-source example

Suppose one token has scalar source values $2$, $5$, and $1$. After key normalization and comparison with the pseudo-query, imagine the logits are $0$, $1$, and $0$. Softmax gives approximate weights $0.212$, $0.576$, and $0.212$.

The retrieved value is

$$
0.212(2) + 0.576(5) + 0.212(1) = 3.516
$$

The second source contributes most, but the embedding and the other layer remain available. If the pseudo-query or normalized source features change, the mixture changes as well.

## Following the source axis

A clean implementation first stacks all sources along a new leading depth axis. RMS-normalize across the final feature dimension, then take the dot product with the pseudo-query. This leaves one logit for every source, batch item, and token.

Softmax must operate over the new source axis. Applying it over tokens would mix unrelated positions, and applying it over features would not produce one weight per depth source. After weighting, sum only over sources. The embedding shape is recovered automatically.

Return both the retrieved representation and the weights. The weights are useful because they reveal how the layer distributed its depth retrieval.

## Implementation order

- Put the embedding first and append the preceding outputs in chronological order.
- Stack those tensors along a source axis.
- RMS-normalize each source only for score computation.
- Compare the normalized keys with the pseudo-query and apply softmax over sources.
- Multiply the original source values by their weights and sum over sources.
- Return the retrieved representation followed by the depth weights.

## Common mistakes to avoid

- **Leaving out the embedding.** It is depth source zero even when earlier layer outputs exist.
- **Normalizing the values.** Only keys are normalized before scoring; retrieval uses raw sources.
- **Using softmax over tokens.** The competition is between depth sources for each token.
- **Giving every token one shared weight vector.** Source content varies by token, so the scores do too.
- **Changing source order.** The returned weights must correspond to the embedding followed by chronological layer outputs.

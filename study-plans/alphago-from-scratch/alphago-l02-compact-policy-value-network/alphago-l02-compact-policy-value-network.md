# Understanding the Compact Policy-Value Network

This problem assembles the complete forward pass of a small AlphaGo Zero style network from supplied parameter tensors. The network turns encoded board history into one policy logit for every action and one bounded value for every position. Its three parts are a trunk, an ordered residual tower, and two output heads.

## The shared trunk

The trunk is the first transformation of the board-history input. It uses a three by three convolution with padding one, so each location can combine nearby information without reducing the board dimensions. The convolution can change the number of channels, creating the shared width used by all residual blocks.

The convolution output is normalized per channel across the batch and board locations. The supplied channel scale and shift are applied, followed by ReLU:

$$
H_0=\operatorname{ReLU}\left(\operatorname{BN}\left(\operatorname{Conv}_{3\times3}(X)\right)\right)
$$

$H_0$ is the shared feature representation that enters the residual tower. The trunk has no separate bias in this contract because the normalization shift already supplies a channel-wise offset.

## Channel normalization

The same normalization rule is reused in the trunk, every residual stage, and both heads. For each channel, compute the mean and population variance over all batch items and spatial locations:

$$
\widehat{z}=\frac{z-\mu}{\sqrt{\sigma^2+\varepsilon}}
$$

Then apply the supplied channel scale and shift:

$$
\operatorname{BN}(z)=\gamma\widehat{z}+\delta
$$

Channels remain separate because each may represent a different learned pattern. The positive constant epsilon makes constant channels safe to normalize. Population variance is required, so the squared deviations are averaged without a sample correction.

Although the rule is shared, every stage receives its own scale and shift and computes statistics from its own activations. Normalization output from one stage cannot be reused elsewhere.

## The ordered residual tower

Each residual parameter group describes one block with two three by three convolutions and two normalization stages. Padding one preserves the feature dimensions throughout the tower.

For block $k$, the first convolution is normalized and passed through ReLU. The second convolution is normalized and then added to the feature tensor that entered that block. ReLU is applied after the addition:

$$
R_k=\operatorname{BN}_2\left(\operatorname{Conv}_2\left(\operatorname{ReLU}\left(\operatorname{BN}_1\left(\operatorname{Conv}_1(H_{k-1})\right)\right)\right)\right)
$$

$$
H_k=\operatorname{ReLU}(H_{k-1}+R_k)
$$

The blocks must be processed in their supplied order because the output of one becomes the input to the next. Each block refines the representation produced so far. All residual convolutions keep the same channel count, which makes shortcut addition possible.

The residual list may be empty. In that case, the trunk output goes directly to the two heads. A correct network should not require at least one block or invent a default block.

## The policy head

The policy head reads the final shared features through its own one by one convolution. A one by one kernel mixes channels at each board location without mixing neighbouring locations. Normalization and ReLU follow, and the remaining channel and board dimensions are flattened within each batch item.

A linear projection converts the flattened policy features into one raw logit for every board intersection plus pass. If the board side length is $n$, each item receives $n^2+1$ logits.

These outputs remain unnormalized. Softmax and legal-action masking belong to MCTS expansion, where probabilities are assigned only to legal edges. Returning raw logits preserves the exact responsibility of this network.

## The value head

The value head has separate parameters because it answers a different question. Its one by one convolution, normalization, and ReLU produce value-specific spatial features. After flattening within each batch item, a hidden linear layer and ReLU combine information from the whole board.

A final linear layer produces one scalar, and tanh bounds it:

$$
v=\tanh(z)
$$

The value is interpreted from the current player's perspective because the encoded input was constructed from that perspective. The trailing single-value dimension is removed so the result contains one scalar per batch item.

## Why parameter groups matter

The residual parameters arrive as an ordered collection, with one group per block. The head parameters arrive as one group containing the policy and value tensors. These collections are part of the problem contract. Each named tensor belongs to one precise transformation.

## Consistent type and device

The entire forward pass must use 32-bit floating point on the input device. Every supplied convolution weight, linear weight, bias, scale, and shift should participate using that same type and device.

This consistency matters across the whole network because output from one stage becomes input to the next. A single parameter left on another device interrupts the chain. Converting tensors for computation should create compatible values without mutating the supplied parameters.

Flattening must preserve the batch dimension. The trunk and residual blocks process all positions together, but each position still needs its own policy vector and value at the end.

## Common mistakes to avoid

- Omitting padding in the trunk or residual tower changes board dimensions and breaks shortcut addition.
- Processing residual groups out of order changes the network being evaluated.
- Applying ReLU before the residual addition prevents negative corrections from reaching the shortcut.
- Treating an empty residual list as an error ignores a valid compact network.
- Applying softmax or legality masking changes the required raw policy output.
- Forgetting tanh leaves the value branch unbounded.
- Mixing parameter groups can produce plausible dimensions but incorrect values.
- Flattening the batch dimension combines independent Go positions.

The assembled network follows one continuous data path: the trunk creates shared spatial features, zero or more residual blocks refine them, and two independent heads read the final representation as action logits and a current-player value.

---
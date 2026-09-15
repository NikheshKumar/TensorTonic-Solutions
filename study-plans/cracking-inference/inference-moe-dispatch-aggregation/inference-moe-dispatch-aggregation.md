# Dispatching Tokens Through Sparse Experts

Top-k routing tells us which experts a token should use and how strongly each expert should contribute. The next task is to turn those routing decisions into an actual Mixture of Experts forward pass.

This involves three stages: send token representations to their selected experts, evaluate the corresponding expert networks, and add the weighted expert results back to the original token positions.

The final output still has one row per input token. A token may temporarily appear in several expert groups, but all of its routed contributions must meet again in its original row.

## Begin with one token

Suppose token 4 is routed to expert 1 with weight $0.7$ and expert 3 with weight $0.3$. Both experts receive the same token representation $x_4$, but they apply different parameters.

If their outputs are $y_{4,1}$ and $y_{4,3}$, the final row for that token is

$$
y_4=0.7y_{4,1}+0.3y_{4,3}
$$

The route weights form a weighted mixture. They are not probabilities used to randomly select one expert. Both selected experts are evaluated, and both outputs contribute.

When $k=1$, there is one selected expert with weight one, so the token output is simply that expert’s output.

## The expert computation

Every expert in this problem is a two-layer feed-forward network without biases. Expert $e$ computes

$$
\operatorname{FFN}_e(x)=\operatorname{ReLU}(xW_{\text{in},e})W_{\text{out},e}
$$

The first matrix expands a token from model width $d_{\text{model}}$ to hidden width $d_{\text{ff}}$. ReLU replaces negative hidden activations with zero. The second matrix projects the hidden vector back to the model width.

Experts share this architecture but not their matrices. Sending a token to the wrong expert means using a different learned function, even though the tensor shapes still look valid.

There are no biases, gates inside the expert, residual connections, or normalization layers in this contract. Adding any of them would compute a different function.

## The tensors involved

The token states have shape $(T,d_{\text{model}})$. The input weights have shape $(E,d_{\text{model}},d_{\text{ff}})$, and the output weights have shape $(E,d_{\text{ff}},d_{\text{model}})$.

Routing produces expert indices and weights with shape $(T,k)$. Token $t$ therefore creates $k$ logical routes, one for each selected expert.

The final result has shape $(T,d_{\text{model}})$ and preserves the original token order.

## Why tokens are grouped by expert

A direct description would loop over tokens and run each of their selected experts. That is correct in meaning, but it misses the main structure of sparse expert execution.

Experts are easier to evaluate in groups. Gather all token routes assigned to expert 0 and process those token states together with expert 0’s matrices. Then do the same for expert 1, and continue through the available experts.

For example, suppose three tokens have routes:

- token 0 goes to experts 1 and 2,
- token 1 goes to experts 0 and 2,
- token 2 goes to experts 2 and 3.

Expert 2 receives token states 0, 1, and 2 as one group. Expert 0 receives only token 1. Expert 1 receives only token 0, and expert 3 receives only token 2.

This grouping is dispatch. It changes the order in which work is performed, but it must not change which token-expert pairs exist.

## A token can appear more than once

With top-k routing, each token is copied logically to $k$ expert groups. The token representation itself is the same on every route, while the expert index and routing weight differ.

If $T=4$ and $k=2$, there are eight route pairs. Some may point to the same expert, and every original token appears exactly twice.

It is useful to keep three aligned pieces of route information:

- the original token index,
- the selected expert index,
- the corresponding routing weight.

Losing this alignment is the central danger in dispatch code. Sorting routes by expert is fine only when token indices and weights follow the same permutation.

## Route weights are applied to expert outputs

For each selected pair $(t,e)$, compute the expert output and multiply it by that route’s weight:

$$
c_{t,e}=w_{t,e}\operatorname{FFN}_e(x_t)
$$

Then sum the contributions belonging to token $t$:

$$
y_t=\sum_{e\in S_t}c_{t,e}
$$

Applying the weight after the expert network matches the contract. Moving it before the expert is generally wrong because ReLU is nonlinear. In particular,

$$
\operatorname{ReLU}(wxW)\ne w\operatorname{ReLU}(xW)
$$

for arbitrary $w$, especially if a weight could change sign. Routing weights here are non-negative, so positive homogeneity of ReLU can make some rearrangements appear to work, but relying on that coincidence obscures the required computation and can fail when the expert definition changes.

## Scatter-add restores token order

After processing an expert group, its rows are arranged by routes to that expert, not by the original sequence. Each weighted result must be added to the output row identified by its stored token index.

This operation is a scatter-add. It scatters routed rows back to token locations and adds when several routes share the same token index.

Ordinary indexed assignment is insufficient for top-k greater than one. The second expert contribution could overwrite the first, leaving only the last route that happened to be processed. Addition is required because the mathematical result is a sum across selected experts.

Initialize the output to zeros. As each expert group finishes, add its weighted rows at the matching token indices. The loop order over experts should not affect the intended result, apart from small floating-point differences caused by addition order.

## Unused experts are valid

Some experts may receive no tokens. That is normal for a particular input and routing result.

An empty expert group should simply contribute nothing. It should not cause a matrix multiplication error, invent dummy tokens, or shift expert numbering. The output still depends on every actual route and only those routes.

The problem does not include expert capacity limits. Every selected route must be processed even if many tokens choose the same expert.

## Routing must match the previous problem

The router behavior is part of this forward pass. For every token, select the top $k$ logits in descending order, let the lower expert index win equal-score ties, and apply softmax only across the selected logits.

Using a different tie rule or normalizing over every expert would make dispatch internally consistent but still wrong. The route weights and indices must match the standalone routing contract exactly.

This is a useful architecture principle: downstream components depend on the precise output contract of upstream components, not merely on shapes that happen to fit.

## Sparse work versus dense work

A dense mixture would evaluate all $E$ experts for all $T$ tokens and then combine them. Sparse top-k execution evaluates only $Tk$ token-expert pairs.

The expert arithmetic is therefore proportional to

$$
O(Tk\,d_{\text{model}}d_{\text{ff}})
$$

rather than the corresponding work with $E$ in place of $k$. Routing still examines the expert logits, and grouping introduces indexing work, but the large feed-forward matrix multiplications follow the number of selected routes.

The logical dispatched representation, route weights, and expert results use memory proportional to $O(TkD)$ for an appropriate feature width $D$. A careful implementation can process one expert group at a time rather than retaining every intermediate expert result.

## Common mistakes to avoid

- Evaluating every expert defeats the sparse computation required by top-k routing.
- Multiplying a route weight into the wrong token-expert pair corrupts the mixture while preserving plausible shapes.
- Assigning outputs back to token rows instead of adding them loses all but one expert contribution.
- Forgetting the original token indices after grouping by expert prevents correct reconstruction of token order.
- Treating an unused expert as an error rejects a valid routing outcome.
- Adding bias terms, residuals, or another activation changes the specified expert function.
- Recomputing routing with a different tie or normalization rule disagrees with the preceding component.

The full operation is best understood as a temporary reorganization. Routing creates token-expert pairs, dispatch groups those pairs so each expert can work efficiently, and scatter-add returns the weighted results to the token rows from which they came.

# Understanding One Policy-Value Training Step

This problem performs one complete optimization step for linear policy and value heads while treating the shared features as fixed. It calculates one pre-update loss, obtains all four parameter gradients from that same parameter state, and then creates a simultaneous SGD update without changing any input tensor.

## The two linear heads

Each batch item has a fixed feature vector. The policy head maps that vector to one logit per action:

$$
L=XW_p^{\mathsf{T}}+b_p
$$

$X$ contains the batch features, $W_p$ contains one row per action, and $b_p$ contains one bias per action. The result contains a policy logit vector for every batch item.

The value head maps the same features to one scalar and applies tanh:

$$
v=\tanh(Xw_v+b_v)
$$

Tanh keeps each prediction between -1 and 1, matching the range of the self-play outcome targets. The value weight is one feature-length vector, and the value bias contains one scalar.

The features are not updated in this exercise. Only the policy weight, policy bias, value weight, and value bias are trainable.

## Policy loss with a soft target

The target policy comes from search visit counts and can distribute probability across several actions. The correct policy term is soft-target cross-entropy.

First apply log-softmax across the action dimension. For each batch item, multiply the resulting log-probabilities by that item's target probabilities, sum across actions, negate, and then average across the batch:

$$
\ell_p=-\frac{1}{B}\sum_{i=1}^{B}\sum_{a=1}^{A}\pi_{i,a}\log p_{i,a}
$$

Log-softmax is numerically stable and keeps the calculation differentiable. Selecting only the largest target action would discard information from the complete search policy.

## Value loss

The value prediction and self-play target contain one scalar per batch item. Their mean squared error is:

$$
\ell_v=\frac{1}{B}\sum_{i=1}^{B}(v_i-z_i)^2
$$

This term trains the value head to predict the eventual outcome from the current player's perspective. Because tanh is part of the forward calculation, its derivative also becomes part of the value-weight and value-bias gradients.

## Regularize all four parameters

The L2 term includes both weights and both biases:

$$
\ell_r=\lambda\left(\lVert W_p\rVert_2^2+\lVert b_p\rVert_2^2+\lVert w_v\rVert_2^2+\lVert b_v\rVert_2^2\right)
$$

Each squared norm is the sum of squared entries. The problem does not exclude biases, average by parameter count, or take a square root.

The regularization coefficient controls the strength of this penalty. When it is zero, regularization contributes no gradient. Otherwise, it pulls every included parameter toward zero in addition to the gradients from its prediction loss.

## One shared pre-update loss

The total loss is calculated at the original parameter values:

$$
\ell=\ell_p+\ell_v+\ell_r
$$

This scalar is the loss returned by the function. It is not recalculated after the SGD update.

All four gradients must come from this one loss and the same original parameter state. If the policy weight were updated before the policy bias gradient was requested, later gradients would refer to a mixed state that never represented one coherent model.

Requesting the gradients together expresses simultaneous optimization. Autograd evaluates:

$$
\nabla_{W_p}\ell,\quad
\nabla_{b_p}\ell,\quad
\nabla_{w_v}\ell,\quad
\nabla_{b_v}\ell
$$

before any updated parameter is constructed.

## The SGD update

Each parameter uses the same rule:

$$
\theta'=\theta-\eta\nabla_{\theta}\ell
$$

$\eta$ is the supplied nonnegative learning rate. A zero learning rate leaves every updated value numerically equal to the original while still requiring the loss and gradients to be computed correctly.

The word simultaneous means that every right-hand side uses an original parameter and its pre-update gradient. Creating a new tensor for each result satisfies this naturally and avoids in-place modifications.

## Protect the caller's tensors

The four supplied parameter tensors must remain unchanged. They may or may not already participate in another computation graph, so the training calculation should use independent 32-bit floating-point leaf copies with gradient tracking enabled.

Detaching, cloning, and then enabling gradients creates local trainable parameters without writing into the originals. It also prevents this one-step function from accidentally connecting its update to an unrelated earlier graph.

The features and targets need 32-bit floating-point interpretation on the same device, but they are not made trainable. All calculations and returned tensors remain on that device.

## Value-bias shape

The single value bias may arrive as a scalar tensor or as a one-element tensor. Internally, treating it as one value makes the linear calculation consistent. The returned updated bias must recover the original shape.

Preserving shape is part of preserving the parameter contract. Two tensors can contain the same scalar but behave differently in later code if one is zero-dimensional and the other has a one-element dimension.

## Detach the outputs

The returned loss and updated parameters should be detached tensors. Their numeric values come from autograd, but the caller receives completed results rather than a graph that continues through the update operation.

Detaching does not convert them to Python numbers or move them to another device.

## Common mistakes to avoid

- Using hard-label cross-entropy discards soft target-policy probabilities.
- Omitting tanh changes both value predictions and their gradients.
- Regularizing only weights ignores the two biases required by the objective.
- Calling backward or gradient extraction separately after changing a parameter breaks simultaneous SGD.
- Updating inputs in place violates the promise that all supplied tensors remain unchanged.
- Returning a post-update loss changes the requested meaning of the first result.
- Losing the original value-bias shape changes the parameter interface.
- Converting the loss to a Python number violates the tensor return contract.
- Leaving outputs attached retains an unnecessary update graph.

One correct step has one consistent timeline: build both predictions from local parameter copies, combine policy, value, and L2 terms into the pre-update loss, obtain all gradients together, and create four detached SGD results from the original values.

---
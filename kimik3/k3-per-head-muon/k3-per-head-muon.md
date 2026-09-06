# Per-Head Muon

Per-Head Muon is an optimizer update for attention projection matrices. It first builds momentum from the current gradient, then orthogonalizes each attention head's row block independently before applying the parameter step. The head partition is the defining part of this problem.

## Momentum comes first

Like other momentum optimizers, Muon keeps a running direction rather than using only the newest gradient. With momentum coefficient $\mu$, compute

$$
M_t = \mu M_{t-1}+G_t
$$

The current gradient is added without an extra factor in the formula used by this exercise. The resulting $M_t$ must be returned because it becomes the previous momentum on the next optimizer step.

Orthogonalization uses this updated momentum. If it is applied to the raw gradient or to the old momentum, the parameter update no longer follows the requested algorithm.

## Why split by attention head

An attention projection stores several heads in one matrix, with equal groups of output rows belonging to different heads. Orthogonalizing the entire matrix at once couples those heads: a large block can affect the normalization of a smaller one.

Per-head Muon separates the output rows into equal contiguous blocks and computes an orthogonalized direction for each block. Every head is therefore scaled and shaped independently.

If the parameter has 12 output rows and three heads, rows 0 through 3 form the first head, rows 4 through 7 form the second, and rows 8 through 11 form the third. The original order must be preserved when the blocks are joined again.

## The polar factor from compact SVD

For one head momentum block, take its compact singular value decomposition:

$$
M_t^{(h)} = U_h\Sigma_hV_h^{\mathsf T}
$$

Discard the singular values and multiply the two orientation factors:

$$
O_h = U_hV_h^{\mathsf T}
$$

This is the polar factor. It keeps the block's principal directions while removing the uneven scale carried by its singular values. For a tall block, its columns are orthonormal; for a wide block, its rows are orthonormal, up to numerical precision.

The problem asks for an exact SVD-based construction. Approximate Newton-Schulz iterations used in large training systems are outside this implementation.

## Apply the update without mutation

Concatenate the head factors in their original row order to form $O$. Then update the parameter with learning rate $\eta$:

$$
\Theta_{t+1}=\Theta_t-\eta O
$$

Return the updated parameter, updated momentum, and orthogonalized update. All three tensors have the same shape as the original parameter.

Do not edit the input parameter or momentum in place. Optimizer logic often retains those tensors elsewhere, and mutation would make the returned values difficult to reason about.

## A diagonal example

Suppose one two-row head has updated momentum

$$
M=\begin{bmatrix}3&0\\0&1\end{bmatrix}
$$

Its singular vectors align with the coordinate axes, while its singular values are $3$ and $1$. The polar factor is the identity matrix:

$$
O=\begin{bmatrix}1&0\\0&1\end{bmatrix}
$$

The large first singular value does not make the first update direction three times larger. Muon keeps orientation but removes that scale imbalance.

If a second head has a different momentum scale, it receives its own SVD and polar factor. This is the practical meaning of per-head processing.

## Implementation order

- Form updated momentum from previous momentum and the current gradient.
- Verify that the number of output rows divides evenly across heads.
- Split updated momentum into contiguous row blocks.
- Run compact SVD on each block and multiply $U_h$ by $V_h^{\mathsf T}$.
- Concatenate the head updates in original row order.
- Subtract learning rate times the combined update from the parameter.
- Return updated parameter, updated momentum, and the combined orthogonalized update.

## Common mistakes to avoid

- **Orthogonalizing the whole matrix.** The operation must be independent for every head block.
- **Using old momentum.** SVD is applied after the current gradient has been incorporated.
- **Keeping the singular values.** The polar factor is $UV^{\mathsf T}$, not the reconstructed original matrix.
- **Splitting columns instead of rows.** Heads occupy equal output-row blocks in this task.
- **Updating in place.** Preserve all supplied tensors.

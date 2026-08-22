# Per-Channel INT8 Quantization

Per-tensor quantization uses one scale for every value. That works well when the tensor has a fairly uniform range, but weight channels can have very different magnitudes. A large channel can force a coarse scale on a much smaller channel.

Per-channel quantization gives every channel its own symmetric INT8 scale. The integer range remains $[-127,127]$, but each channel gets a ruler sized for its own values.

This problem extends the preceding symmetric scheme. The main new ideas are choosing a channel axis, reducing every other axis, and keeping the scales shaped so they broadcast back over the input.

## Why separate scales help

Consider two rows:

$$
\begin{bmatrix}
0.1 & -0.2 & 0.3\\
40 & -80 & 20
\end{bmatrix}
$$

One scale for the entire matrix is controlled by 80. Its step is about $80/127\approx0.63$, so every value in the first row rounds to zero.

If each row is treated as a channel, the first row uses $0.3/127$, while the second uses $80/127$. Both rows can use most of the available code range, even though their magnitudes differ by hundreds of times.

The benefit comes from local range selection, not from changing the number of integer codes.

## What the channel axis means

The channel axis identifies which positions receive independent scales. Every index along that axis defines one channel.

For a matrix with shape $(R,C)$:

- channel axis 0 gives one scale per row,
- channel axis 1 gives one scale per column.

For a rank-three tensor, the same rule applies. If the shape is $(A,B,C)$ and the channel axis is 1, there are $B$ channels. Each channel contains all values across axes 0 and 2 at its fixed axis-1 index.

The channel axis is preserved. Every other axis is reduced when finding channel maxima.

## Calculating each scale

For channel $c$, find the largest absolute value among only that channel’s elements:

$$
a_c=\max_{i\text{ inside channel }c}|x_i|
$$

For a nonzero channel, its scale is

$$
s_c=\frac{a_c}{127}
$$

Each element uses the scale belonging to its channel:

$$
q_i=\operatorname{clip}\left(\operatorname{round}\left(\frac{x_i}{s_{c(i)}}\right),-127,127\right)
$$

where $c(i)$ is the element’s channel index.

There is no sharing across channels during maximum calculation. A large value in channel 4 must not change the scale for channel 3.

## Scale shapes and broadcasting

The scale tensor keeps the channel dimension at its original size and gives every reduced dimension size one.

For an input of shape $(2,3)$:

- axis 0 scales have shape $(2,1)$,
- axis 1 scales have shape $(1,3)$.

For shape $(2,3,4)$ with channel axis 1, scales have shape $(1,3,1)$.

These singleton dimensions let ordinary broadcasting apply each scale to the correct channel. A $(1,3,1)$ scale expands across the first and last axes without copying one channel’s scale into another.

Dropping the reduced dimensions can create a one-dimensional tensor of length three, but that tensor does not necessarily align with the intended axis during broadcasting. Preserving dimensions makes the relationship explicit and reliable.

## Row-wise and column-wise examples

Take

$$
X=\begin{bmatrix}
1 & 2 & 4\\
10 & 20 & 40
\end{bmatrix}
$$

With channel axis 0, the row maxima are 4 and 40. The scales are $4/127$ and $40/127$, stored with shape $(2,1)$.

With channel axis 1, the column maxima are 10, 20, and 40. The scales are $10/127$, $20/127$, and $40/127$, stored with shape $(1,3)$.

Both are valid per-channel schemes, but they answer different definitions of channel. The supplied axis decides which one the function must implement.

## Negative axis notation

Tensor axes can also be written from the end. For a rank-three tensor:

- axis $-1$ means axis 2,
- axis $-2$ means axis 1,
- axis $-3$ means axis 0.

Normalizing a negative axis produces the equivalent non-negative axis before reduction axes are chosen.

For example, channel axis $-1$ and channel axis 2 must produce identical results on the same rank-three input. Treating $-1$ as a separate or invalid direction would violate ordinary tensor-axis semantics.

## Zero channels need local handling

One channel may be entirely zero while other channels contain normal values. Its absolute maximum is zero, so its normal scale formula would divide by zero.

That channel receives fallback scale one and zero codes. Other channels still use their own maximum-derived scales.

The fallback must be applied per channel. Checking whether the complete tensor is zero misses the case where one row or column is zero inside an otherwise nonzero tensor.

The result remains finite, and dequantization of the zero channel stays zero.

## Dequantization and error

Reconstruction uses the same broadcasted channel scales:

$$
\hat{x}_i=q_i s_{c(i)}
$$

Each output element must equal its integer code converted to floating point and multiplied by the matching channel scale.

Smaller-range channels usually get smaller scale steps, which reduces their rounding error compared with a global scale controlled by another channel’s outlier.

This does not guarantee exact recovery. Values still map to a finite integer grid, and rounding still loses information.

## Granularity tradeoff

Per-channel quantization stores more metadata than per-tensor quantization because it needs one scale per channel rather than one scale for the complete tensor.

The extra scales can improve accuracy when channel ranges differ. The tradeoff is a slightly more complex representation and broadcasting pattern.

This exercise stops at channel granularity. The next problem divides each row into even smaller groups, allowing different column regions within one row to have independent scales.

## Cost and memory

Every element contributes to one channel maximum and then undergoes quantization and reconstruction, so the running time is $O(N)$ for $N$ tensor elements.

Codes and reconstruction use $O(N)$ storage. Scale metadata uses $O(C)$ values for $C$ channels, represented with singleton dimensions for broadcasting.

The algorithm works for rank-two and rank-three tensors without assuming rows are always the channel dimension.

## Common mistakes to avoid

- Reducing along the channel axis produces one scale for each position in the wrong dimensions.
- Reducing only one non-channel axis in a rank-three tensor leaves scales dependent on an extra dimension.
- Removing reduced dimensions can align scales with the wrong axis during broadcasting.
- Using one global zero check fails when only a single channel is zero.
- Letting one channel’s maximum affect another defeats per-channel quantization.
- Rejecting a valid negative axis breaks equivalence with its normalized axis.
- Using the full INT8 range down to $-128$ violates the symmetric $[-127,127]$ contract.
- Returning a one-dimensional scale vector when a broadcastable keep-dimension shape is required loses the stated shape contract.

The central operation is selective reduction: preserve the chosen channel axis, summarize every other axis, and broadcast the resulting scales back to the values that created them.

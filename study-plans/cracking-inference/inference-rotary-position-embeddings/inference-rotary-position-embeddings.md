# Rotary Position Embeddings for Decoding

Attention compares token content, but content alone does not tell the model where a token appears. If the same word occurs at positions 3 and 300, its unpositioned query and key features could look identical. The model needs some representation of order before it can distinguish “the dog chased the cat” from “the cat chased the dog.”

Rotary Position Embeddings, usually called RoPE, add position information by rotating query and key features. Nothing is appended to the vector, and no position vector is added. Instead, adjacent features are treated as two-dimensional points and turned by an angle determined by the token’s position.

## Begin with an ordinary rotation

Take a two-dimensional point $(x_0,x_1)$. Rotating it by angle $\phi$ produces

$$
x'_0=x_0\cos\phi-x_1\sin\phi
$$

$$
x'_1=x_0\sin\phi+x_1\cos\phi
$$

You can picture the original pair as the hand of a clock. Rotation changes the direction of the hand while preserving its length. RoPE applies exactly this operation to every adjacent even-odd feature pair in a query or key vector.

For a vector $[x_0,x_1,x_2,x_3]$, the pairs are $(x_0,x_1)$ and $(x_2,x_3)$. Each pair rotates independently. This is why the head width must be even: every feature needs a partner.

## Position controls the angle

The rotation angle is the token position multiplied by a frequency. If $p$ is the absolute position and pair $i$ has inverse frequency $\theta_i$, then

$$
\phi_{p,i}=p\theta_i
$$

The frequency schedule used by this problem is

$$
\theta_i=\operatorname{base}^{-2i/d_k}
$$

where $d_k$ is the head width and $i$ runs over the feature pairs.

Different pairs rotate at different speeds. Early pairs use higher frequencies and turn quickly as position changes. Later pairs use lower frequencies and turn slowly. Together, these rotations give the attention mechanism several scales on which to represent position.

The base, commonly $10000$, controls how the frequencies are spread. It is not a learned value in this exercise.

You do not need to memorize the frequency formula before understanding the purpose it serves. A single rotation speed would make every pair repeat on the same cycle, which would give the model a narrow view of distance. The frequency schedule creates a mixture of short and long cycles, much like using the second, minute, and hour hands of a clock to describe time at different scales.

## A small example

Suppose a feature pair is $[1,0]$ and its inverse frequency is $1$. At position zero, the angle is zero, so

$$
[1,0]\longrightarrow[1,0]
$$

At position one, the angle is one radian, giving approximately

$$
[1,0]\longrightarrow[0.5403,0.8415]
$$

At position two, the same pair rotates to approximately

$$
[-0.4161,0.9093]
$$

The content vector started the same each time, but its direction now carries position information. A lower-frequency pair would move more slowly across those positions.

Position zero is an especially useful test. Every angle is zero, cosine is one, and sine is zero, so both queries and keys must remain unchanged.

## Why rotate queries and keys?

Attention scores come from query-key dot products. When a query at position $p$ and a key at position $q$ are both rotated with the same frequency schedule, their dot product depends on the difference $p-q$.

The geometric reason is simple. Rotating both vectors by the same amount does not change their relative angle. Only the difference between their rotations matters. If one token moves five positions farther away, the query-key comparison changes according to that five-position offset.

RoPE therefore uses absolute positions to create a useful relative-position effect inside attention. The function still receives the absolute position of every sequence slot, and it must use those supplied values exactly.

Values are not rotated in this problem. Position changes which keys a query matches, while the values remain the content to be combined after attention weights are computed.

## Why explicit positions matter during decoding

During full-sequence processing, positions often look like $0,1,2,\ldots$. It is tempting to generate that range from the local sequence length. That fails during incremental decoding.

Suppose a model has already processed 128 tokens and now receives a small query/key slice for positions 128 and 129. The local slice indices are 0 and 1, but their true positions are 128 and 129. Rotating them as 0 and 1 would make the model treat them as the beginning of a sequence.

This exercise supplies a position tensor precisely to avoid that mistake. Positions may begin at any non-negative offset, and they do not need to be contiguous. The same positions are shared across batch items and heads, but each sequence slot uses its own supplied value.

## Rotation preserves information length

A two-dimensional rotation is an orthogonal transformation, so

$$
(x'_0)^2+(x'_1)^2=x_0^2+x_1^2
$$

Applying the operation pair by pair preserves the norm of the entire query or key vector. RoPE changes direction, not magnitude.

Norm preservation is more than a mathematical curiosity. It gives you a strong implementation test. If rotated vectors become consistently longer or shorter, the sine and cosine terms were probably paired or signed incorrectly.

## Following the tensors without losing the idea

Queries and keys have shape $(B,H,S,d_k)$, representing batch, head, sequence position, and head features. The positions have shape $(S)$.

The angle grid needs one row per sequence position and one column per feature pair. Its natural shape is $(S,d_k/2)$. Sine and cosine values from that grid broadcast across the batch and head dimensions.

For both Q and K:

1. take the even features and odd features,
2. apply the two rotation equations using the same angles,
3. place the rotated values back into their original even and odd locations.

The returned tensors retain the input shapes and dtypes. No head or sequence position should be mixed with another.

## Cost and memory

Every query and key feature is touched a constant number of times, so the work is $O(BHSd_k)$. The position-frequency grid uses $O(Sd_k)$ values before broadcasting, while the rotated outputs have the same size as their inputs.

RoPE does not form an attention matrix. It prepares queries and keys before their dot products are computed.

## Common mistakes to avoid

- Using local indices instead of the supplied absolute positions breaks offset decoding.
- Rotating the first half against the second half implements a different pairing from adjacent even-odd features.
- Giving the two features in a pair different frequencies breaks the planar rotation.
- Reversing a sine sign changes the rotation direction and can destroy the relative-position behavior.
- Applying RoPE to values changes the contract.
- Forgetting that $d_k$ must be even leaves an unpaired feature.

The most useful mental picture is a collection of clock hands. Position turns each hand at its own speed, and attention reads the relative angle between query and key hands.

# Sinusoidal Timestep Embedding

A diffusion model removes noise a little at a time, and at every step it has to know how noisy its current input is. That noise level arrives as a single number called the timestep, which is an awkward thing for a neural network to read directly. A sinusoidal embedding turns that number into a list of smooth waves, a bit like reading the time from a clock that has a fast second hand, a slower minute hand and a very slow hour hand all at once. This page builds the idea one wave at a time, starting from a single cosine and ending with the full vector of cosines and sines that you compute here for every timestep in a batch.

## Why the network cannot use the raw timestep

The same network weights are shared across every noise level. At an early timestep the input is almost clean and the network should make small corrections, while at a late timestep the input is almost pure noise and the network has to invent most of the structure. The only way the network can behave differently at these two extremes is if the timestep reaches it as an input.

Feeding the raw timestep $t$ as one extra number works poorly for three reasons. Its scale runs from 0 to about 1000, far larger than normalized activations. A single scalar gives the next linear layer only one direction to work with. And after normalization, neighbouring timesteps such as 400 and 401 become almost the same number. The fix is a code that is bounded, smooth in $t$ and spread over many dimensions.

## One wave gives a bounded code

The simplest bounded code is a single cosine, $\cos(t)$. Every value lies between $-1$ and $1$ however large $t$ gets, and small changes in $t$ give small changes in the code, so the network receives a well behaved number.

A single cosine has two problems. It is symmetric around each peak, so $t = 1$ and $t = 2\pi - 1 \approx 5.2832$ both give $\cos t \approx 0.5403$. It also repeats every $2\pi \approx 6.28$ timesteps, so $t = 0$ and $t = 44$ look almost identical, since $\cos 44 \approx 0.99984$ is very close to $\cos 0 = 1$.

## A sine beside the cosine fixes the symmetry

Pair the cosine with a sine at the same frequency. The two numbers $(\cos t, \sin t)$ are a point on a circle, and the angle of that point is recovered exactly. For $t = 1$ the sine is about $0.8415$, while for $t = 2\pi - 1$ it is about $-0.8415$, so the two timesteps that collided before are now told apart.

The pair has a second useful property. By the angle addition formulas, the pair at $t + k$ is a fixed rotation of the pair at $t$, and a rotation is a linear map. A linear layer can therefore express "a little later" or "a little earlier" as one fixed transformation, which makes the code friendly to relative offsets.

The repetition problem remains. At $t = 44$ the pair is about $(0.99984, 0.0177)$, almost the same point as $(1, 0)$ at $t = 0$.

## Slower waves separate distant timesteps

Repetition comes from the speed of the wave. Multiply $t$ by a frequency $f$ before taking the cosine and sine, so the angle is $ft$ and the wave repeats every $2\pi / f$ timesteps. With $f = 0.01$, the pair at $t = 44$ is about $(0.9048, 0.4259)$, clearly far from $(1, 0)$.

A slow wave has the opposite weakness. With $f = 0.0001$, the cosine at $t = 400$ is about $0.999200$ and at $t = 401$ it is about $0.999196$, a difference too small to matter. Fast waves resolve neighbours but repeat, and slow waves never repeat in range but blur neighbours together, so the code needs both kinds at once.

## A ladder of frequencies

The embedding stacks several cosine and sine pairs, each at its own frequency. With $\texttt{half}$ frequencies, index $i$ runs from $0$ to $\texttt{half} - 1$ and the frequencies form a geometric ladder,

$$
f_i = \exp\left(-\ln(\texttt{max\_period}) \cdot \frac{i}{\texttt{half}}\right) = \texttt{max\_period}^{-i/\texttt{half}} .
$$

Here max_period is the number that sets the slowest wave. The first frequency is exactly 1, and each next frequency is smaller by the constant ratio $\texttt{max\_period}^{-1/\texttt{half}}$. The frequencies are evenly spaced on a logarithmic scale, and the last one is a little larger than $1 / \texttt{max\_period}$, since the exponent stops one step short of $-1$.

With max_period 10000 and three frequencies, the ratio is about $0.046416$, the frequencies are $1$, about $0.046416$ and about $0.0021544$, and the periods are about 6.28, 135.4 and 2916.4 timesteps. The fast pair separates neighbouring timesteps, the slow pair never repeats within 1000 steps and so separates early from late, and the middle pair covers the scales in between.

Together the pairs behave like the digits of a number written in a smooth, continuous base. Fast columns resolve fine detail and slow columns resolve the coarse position. Geometric spacing is what lets a handful of columns cover timescales that differ by a factor of thousands, which evenly spaced frequencies could not do without wasting most of their columns on similar speeds.

## Laying out the vector for a batch

For a sample $b$ with timestep $t_b$, every frequency gives an angle $a_{b,i} = t_b f_i$. The row for that sample lists all $\texttt{half}$ cosines first, in frequency order, and then all $\texttt{half}$ sines in the same order, so its length is $2 \cdot \texttt{half}$. This cos-then-sin block layout is a fixed convention that the rest of the network relies on.

The requested length is dim, and $\texttt{half}$ is dim divided by 2 with the remainder dropped. When dim is even the two blocks fill the row exactly. When dim is odd they leave one slot empty, so a single zero is appended at the end. That zero carries no information and exists only so the vector matches the width of the linear layer that reads it.

A batch of B timesteps produces a matrix of shape (B, dim), one row per sample. The frequencies are the same for every row, and only the angles change with $t_b$.

## Similar timesteps get similar vectors

Because each frequency contributes a cosine and a sine, the dot product of two embeddings depends only on the difference of the timesteps. For one frequency, $\cos(a)\cos(c) + \sin(a)\sin(c) = \cos(a - c)$, and summing over frequencies keeps that property.

With dim 16 and max_period 10000, the embedding of $t = 5$ has dot product 8 with itself, about 7.49 with $t = 6$, about 4.26 with $t = 50$ and about 2.88 with $t = 500$. Nearby timesteps receive similar vectors and distant ones receive different vectors, which is exactly the smoothness a denoiser needs.

## A five-dimensional example

Take a single timestep $t = 4$ with dim 5 and max_period 10000.

1. $\texttt{half}$ is 5 divided by 2 with the remainder dropped, which is 2.
2. The frequencies are $f_0 = 1$ and $f_1 = 10000^{-1/2} = 0.01$.
3. The angles are $4 \cdot 1 = 4$ and $4 \cdot 0.01 = 0.04$.
4. The cosines are $\cos 4 \approx -0.653644$ and $\cos 0.04 \approx 0.999200$.
5. The sines are $\sin 4 \approx -0.756802$ and $\sin 0.04 \approx 0.039989$.
6. The row is about $(-0.653644, 0.999200, -0.756802, 0.039989, 0)$.

The fast pair has already wrapped past half a turn, while the slow pair sits near $(1, 0.04)$, with the sine almost equal to the angle itself. That shows how little the slow frequency moves at small timesteps. The final zero is the padding for odd dim.

## Precision and fractional timesteps

Large angles are the main precision concern. At $t = 999$ the fastest angle is 999 radians, and float32 stores numbers of that size with a spacing of about $6 \times 10^{-5}$, so a rounding error of that size goes straight into the cosine and sine. This is harmless for a network input, but it is why two correct float32 implementations can differ in the fourth or fifth decimal place at large timesteps.

Timesteps can also be fractional, for example in continuous-time samplers, and the formula works unchanged for float inputs. Whatever the type of the incoming timesteps, the embedding is returned in float32.

## Where this sits in the denoising network

The embedding has no learnable parameters. In a DDPM U-Net (Ho et al., 2020) it is passed through a small MLP of two linear layers with a SiLU between them, and the result is added to the feature maps inside every residual block. The construction itself comes from the positional encodings of the Transformer (Vaswani et al., 2017), with the timestep playing the role of a token position.

## Common mistakes to avoid

- **Putting the sines before the cosines.** The column order is a fixed convention, and later layers and pretrained weights expect the cosine block first.
- **Interleaving cosine and sine columns.** Alternating cos, sin, cos, sin produces the right numbers in the wrong order, so the next linear layer sees permuted features.
- **Dividing the frequency index by half minus 1.** That shifts every frequency except the first and makes the slowest one exactly 1 / max_period, which no longer matches the standard ladder.
- **Hardcoding 10000 as the period.** The max_period argument sets the whole ladder, and ignoring it gives wrong angles for every pair after the first.
- **Forgetting the zero column for odd dim.** The row comes out one entry short and no longer matches the input width of the time MLP.
- **Building the frequencies in float64.** A float64 frequency tensor promotes the whole embedding to float64, while the result must always be float32.
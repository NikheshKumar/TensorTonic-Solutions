# Time-Conditioned Residual Block

Most of the work inside a diffusion denoiser is done by a repeated unit called a residual block. It looks at a stack of feature maps, works out a small correction and adds that correction back to what it was given, much like an editor who marks a few changes on a draft and keeps the rest of the text. A diffusion model also needs every block to know how noisy the image is, so each block reads a description of the timestep and uses it to nudge its features up or down. This page builds the block piece by piece, starting from a residual update on one number and ending with the full block, with its normalization, activation, convolutions, time signal and skip path, that you implement here.

## A residual update keeps the input and adds a correction

Start with one number, a feature value $x = 1.5$. A plain layer would replace it with some new value. A residual layer instead computes a correction from $x$ and adds it to $x$, so if the correction is $-0.3$ the output is $1.5 + (-0.3) = 1.2$.

This matters for two reasons (He et al., 2016). The branch that computes the correction only has to learn what should change, and a branch that outputs zero leaves the input untouched, which is an easy starting point. Gradients also reach early layers through the plain addition without passing through every weight, so deep stacks of blocks train reliably.

For a feature map the same idea holds pixel by pixel. The block output is the input plus a correction branch of the same shape,

$$
\text{out} = \mathrm{skip}(x) + \text{branch}(x),
$$

where for now $\mathrm{skip}(x)$ is simply $x$ itself. The rest of this page is about what goes inside the branch and how the timestep enters it.

## Why every block needs to know the timestep

One network denoises inputs at every noise level, from almost clean images to almost pure noise. The right amount of smoothing, sharpening and invention differs enormously between those extremes. If the timestep entered only at the first layer, its influence would fade as the signal passes through many layers, so diffusion U-Nets feed it again into every residual block (Ho et al., 2020).

Think of one output channel first. The timestep arrives as an embedding vector $e$ of length E, usually a sinusoidal code refined by a small MLP. The block turns that vector into a single number for the channel by taking a weighted sum of its entries and adding a bias. That number is added to every pixel of the channel, so the whole channel shifts up or down by an amount that depends on the noise level.

The embedding first passes through SiLU, the activation $\mathrm{SiLU}(z) = z \cdot \mathrm{sigmoid}(z)$. SiLU is a smooth relative of ReLU that lets small negative values through, so $\mathrm{SiLU}(1) \approx 0.7311$ and $\mathrm{SiLU}(-2) \approx -0.2384$.

## From one channel to every channel

With C_out output channels, the block needs one shift per channel, and it gets them all from one linear layer. Here $e$ is time_embedding of shape (B, E), $W_t$ is time_weight of shape (C_out, E) and $b_t$ is time_bias of shape (C_out,),

$$
\tau = \mathrm{SiLU}(e)\, W_t^{\top} + b_t .
$$

Row $c$ of $W_t$ holds the weights for channel $c$, which is the standard linear-layer layout, so $\tau$ has shape (B, C_out). The value $\tau_{b,c}$ is added to every pixel of channel $c$ in sample $b$. The time signal therefore acts as a learned per-channel bias that changes with the noise level, and the same embedding can push different channels in different directions.

## Group norm normalizes one sample at a time

Before each convolution the block normalizes its input. Start with one channel of a 2x2 map holding 1, 2, 3 and 4. Its mean $\mu$ is 2.5 and its biased variance $\sigma^2$, the average squared distance from the mean, is 1.25. Each value becomes $(z - \mu) / \sqrt{\sigma^2 + 10^{-5}}$, which gives about $-1.3416$, $-0.4472$, $0.4472$ and $1.3416$, and then a per-channel weight and bias rescale and shift the result.

Batch norm would compute these statistics across the batch, which is a poor fit for diffusion because each sample in a batch sits at a different noise level and batches are often small. Group norm (Wu and He, 2018) computes them for each sample separately. It splits the channels into num_groups groups of consecutive channels and pools each group's channels and pixels together.

The group size changes what is kept. Take two channels, one holding 1, 2, 3 and 4 and one holding 11, 12, 13 and 14. With two groups each channel is normalized on its own and both become the same four values, $\pm 1.3416$ and $\pm 0.4472$. With one group the eight values share a mean of 7.5 and a variance of 26.25, so the first channel becomes about $-1.27$ to $-0.68$ and the second about $0.68$ to $1.27$, and the fact that the second channel sat higher survives.

## The order of the operations

The block uses the pre-activation style, where group norm and SiLU come before each 3x3 convolution. The first half reads the input, and the time shift is added right after its convolution,

$$
h = \mathrm{conv}_1\big(\mathrm{SiLU}(\mathrm{GN}_1(x))\big) + \tau .
$$

Here $x$ has shape (B, C_in, H, W), $\mathrm{GN}_1$ and $\mathrm{conv}_1$ use the norm1 and conv1 entries of params, and $h$ has shape (B, C_out, H, W). The second half repeats the pattern on $h$ and adds the skip path,

$$
\text{out} = \mathrm{skip}(x) + \mathrm{conv}_2\big(\mathrm{SiLU}(\mathrm{GN}_2(h))\big).
$$

Both convolutions use stride 1 and padding 1, which keeps the height and width unchanged so the final addition lines up. Normalizing in front of each convolution means the branch always receives well scaled inputs, while the skip path carries the unnormalized signal straight through. The time shift sits between the two halves, so the second convolution can mix it with neighbouring pixels and channels.

## Why the group size decides whether time matters

The time shift is added immediately before the second group norm, and that norm subtracts a mean. If each group holds one channel, adding a constant to every value of that channel moves its mean by exactly that constant, so normalization removes the shift completely and the time embedding has no effect at all.

With several channels per group, the shift differs between channels in the same group, so it changes their relative levels and survives normalization. Take four channels of 2x2 values, (1, 2, 3, 4), (0, -1, 2, 1), (3, 3, 1, 0) and (-2, 0, 1, 2), and add the per-channel shift (1, -1, 0.5, 0). With four groups the group norm output does not change at all, with two groups it changes by up to about 0.55 and with one group by up to about 0.67. This is one reason real U-Nets use 32 groups over hundreds of channels instead of one group per channel.

## When the skip path needs its own weights

The skip path adds $x$ back to the branch output, which only works when both have the same number of channels. When the block changes the channel count, a 1x1 convolution with skip_weight of shape (C_out, C_in, 1, 1) and skip_bias of shape (C_out,) projects $x$ to C_out channels first. A 1x1 convolution is a per-pixel linear map across channels and does not mix neighbouring pixels.

A learned 1x1 skip is also allowed when C_in equals C_out, so the presence of skip_weight in params decides which path is used. When it is absent, the skip is the identity and $x$ is added unchanged.

## A small numerical example

Take a time embedding $e = (1, -2)$, a time projection with rows $(0.5, 0.25)$ and $(-1, 0)$, and bias $(0.1, 0)$.

1. SiLU gives $\mathrm{SiLU}(1) \approx 0.73106$ and $\mathrm{SiLU}(-2) \approx -0.23841$.
2. Channel 0 receives $0.5 \cdot 0.73106 + 0.25 \cdot (-0.23841) + 0.1 \approx 0.36553 - 0.05960 + 0.1 \approx 0.40593$.
3. Channel 1 receives $-1 \cdot 0.73106 + 0 \cdot (-0.23841) + 0 \approx -0.73106$.

Every pixel of channel 0 in the first convolution's output moves up by about 0.406, and every pixel of channel 1 moves down by about 0.731. If these two channels formed one group in the second group norm, the gap of about 1.137 between their shifts would change their relative levels and reach the output. If each formed its own group, the norm would cancel both shifts.

## Edge cases and numerical care

The variance in group norm is the biased one, dividing by the number of elements, and the eps of $10^{-5}$ keeps the division finite when a group is constant. A 1x1 image with one channel per group is such a case, and its normalized values are exactly 0 before the affine bias. The block must return the dtype of $x$, and float64 inputs should stay float64 throughout.

## Where this block sits in the U-Net

Every resolution level of a diffusion U-Net stacks one or more of these blocks, on the way down, at the bottom and on the way up. The blocks on the way up receive concatenated features from the encoder, so their channel count changes and they use the 1x1 skip convolution.

## Common mistakes to avoid

- **Ignoring num_groups and normalizing with a single group.** This turns group norm into a layer norm over the whole sample and changes the output whenever more than one group is requested.
- **Skipping the SiLU on the time embedding.** The projection then sees the raw embedding, so every channel receives the wrong shift.
- **Adding the time shift in the wrong place.** The shift belongs after the first convolution and before the second group norm, and moving it changes how the rest of the block transforms it.
- **Applying SiLU before group norm.** The activation then acts on unnormalized features, and the result differs from the standard pre-activation block.
- **Choosing the skip path by comparing channel counts.** The presence of skip_weight decides the path, and a learned 1x1 skip is possible even when C_in equals C_out.
- **Returning only the residual branch.** Without adding skip(x) the block loses its identity path, and the output no longer contains the input signal.
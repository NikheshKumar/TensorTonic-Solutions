# Tiny U-Net Forward Pass

Almost every image diffusion model uses a network called a U-Net to guess the noise in an image. It works a little like studying a photo up close, stepping back across the room to see the overall shapes, and then leaning in again while keeping the notes from the first close look so no fine detail is lost. Drawn on paper, that path down to a coarse view and back up has the shape of a U, which gives the network its name. This page builds the network one idea at a time, from why it changes resolution at all, through the down and up steps and the skip connection, to the time signal that reaches every block, ending with the tiny end-to-end U-Net you wire here.

## What a denoiser has to output

A diffusion denoiser receives a noisy image and its timestep and predicts the noise that was added. That prediction has one value per pixel and channel, so the output must have the same height and width as the input. With $x$ of shape (B, C_in, H, W), the output has shape (B, C_out, H, W), where C_out is usually equal to C_in for noise prediction.

The network also has to reason about large structures, such as whether a blurry shape is a face or a cloud. A stack of 3x3 convolutions at full resolution sees only a small window. One layer sees 3 pixels across, two layers see 5, and $L$ layers see $2L + 1$. For a corner pixel of a 64 by 64 image to see the opposite corner, the window must span 127 pixels, which takes 63 layers. That is far too slow a way to gather context.

## Halving the resolution widens the view

The fix is to work at a coarser resolution for part of the network. A 3x3 convolution with stride 2 and padding 1 moves its window two pixels at a time, so an input with an even number of rows $H$ gives

$$
\left\lfloor \frac{H + 2 - 3}{2} \right\rfloor + 1 = \frac{H}{2}
$$

rows, and the same holds for the columns. After one halving, each pixel of the coarse map summarizes roughly a 2x2 patch of the original, and every later 3x3 convolution covers twice as much of the image as it would at full resolution.

Take a 4x4 map holding 1 to 16 row by row and a 3x3 kernel whose nine weights are all one ninth. At stride 2 and padding 1 the windows are centred on pixels (0, 0), (0, 2), (2, 0) and (2, 2), and the 2x2 result is about 1.556, 3.333, 6.333 and 11. The first value is small because its window hangs over the top-left corner, where five of the nine positions are zero padding, so only 1, 2, 5 and 6 contribute and their sum of 14 is divided by 9.

A strided convolution learns how to summarize each patch, which makes it a learned version of downsampling. The U-Net of Ronneberger et al. (2015) used pooling for this step, while DDPM models (Ho et al., 2020) use the strided convolution.

## Coming back up to full resolution

The coarse map now has to grow back to the input size. The simplest way is nearest-neighbour repetition, which copies each pixel into a 2x2 block. On the result above, the top-left quarter of the new 4x4 map holds about 1.556 four times and the bottom-right quarter holds 11 four times.

Repetition alone produces blocky features with hard edges between the copied blocks. A 3x3 convolution with padding 1 applied after the repetition lets the network smooth and sharpen those features with learned weights, so the up step is repetition followed by a convolution.

## The skip connection restores the fine detail

Halving the resolution throws away detail that the coarse map cannot give back. Repeating its pixels only spreads the coarse values around. The U-Net therefore saves the full-resolution features from before the down step, here called $h_1$, and brings them back after the up step.

The saved features and the upsampled features $u$ are joined along the channel axis, written $[u, h_1]$, with $u$ first. With C channels each, the joined map has 2C channels and carries both the global context from the bottom and the exact local detail from the top. Concatenation keeps the two signals in separate channels and lets the next convolution decide how to combine them, which is more flexible than adding them into the same channels. The order matters too, since the weights that read the joined map expect the upsampled channels in the first C slots.

Because the channel count is now 2C, the residual block that reads the joined map cannot add its input directly to its C-channel output. Its skip path is a 1x1 convolution with up_res_skip_weight of shape (C, 2C, 1, 1), which maps the 2C channels back to C at every pixel.

## Every block knows the timestep

The same network handles every noise level, so the timestep has to reach every residual block. It enters as a sinusoidal vector of length E, the cosines and then the sines of $t f_i$ with

$$
f_i = 10000^{-2i/E}, \qquad i = 0, \dots, E/2 - 1 .
$$

This is the same frequency ladder as $\exp(-\ln(10000) \cdot i / (E/2))$, so with E equal to 4 the frequencies are 1 and 0.01. A small MLP of two linear layers refines this fixed code into a learned one,

$$
e = \mathrm{lin}_2\big(\mathrm{SiLU}(\mathrm{lin}_1(\mathrm{emb}(t)))\big),
$$

where $\mathrm{lin}_1$ uses time_linear1_weight of shape (T, E), $\mathrm{lin}_2$ uses time_linear2_weight of shape (T, T), and $e$ has shape (B, T). The SiLU between the layers matters, because without it the two linear layers collapse into a single linear map of the embedding.

The same vector $e$ goes to all three residual blocks. Each block has its own P_time_weight of shape (C, T), so each one projects $e$ into its own per-channel shifts and decides for itself how the noise level should change its features.

## Each residual block in the network

Every block marked R follows one pattern. It applies group norm, SiLU and a 3x3 convolution, adds the projection of $\mathrm{SiLU}(e)$ to every pixel of each channel, applies group norm, SiLU and a second 3x3 convolution, and adds a skip path. Every group norm uses num_groups groups and eps $10^{-5}$.

The blocks differ only in what they read. The down block and the middle block read C channels and use the input itself as the skip. The up block reads the 2C joined channels and uses the 1x1 skip convolution described above. The table of params names them down_res, mid_res and up_res.

## The whole wiring

With every piece in place, the full network reads

$$
h_1 = \mathrm{R}_{\text{down}}(\mathrm{conv\_in}(x), e), \qquad u = \mathrm{up}\big(\mathrm{R}_{\text{mid}}(\mathrm{down}(h_1), e)\big),
$$

$$
\text{out} = \mathrm{conv\_out}\Big(\mathrm{SiLU}\big(\mathrm{GN}(\mathrm{R}_{\text{up}}([u, h_1], e))\big)\Big).
$$

The input convolution conv_in lifts the C_in image channels to C feature channels at full resolution, and the down block refines them into $h_1$. The output head is a final group norm, a SiLU and a 3x3 convolution that maps C features to C_out channels. Normalizing before this last convolution keeps the prediction on a consistent scale across timesteps.

## Following the shapes through the network

Take one RGB image of 6x6 pixels with C equal to 4 and C_out equal to 3, so $x$ has shape (1, 3, 6, 6).

1. The input convolution and the down block give (1, 4, 6, 6), and this is $h_1$.
2. The stride-2 convolution gives $\lfloor (6 + 2 - 3) / 2 \rfloor + 1 = 3$ rows and columns, so (1, 4, 3, 3).
3. The middle block keeps (1, 4, 3, 3).
4. Repetition and the up convolution give (1, 4, 6, 6), and this is $u$.
5. Joining $u$ and $h_1$ gives (1, 8, 6, 6).
6. The up block, with its 1x1 skip, returns to (1, 4, 6, 6).
7. The output head gives (1, 3, 6, 6), the shape of the input.

Every pixel of the middle block sees a 3x3 window of a map that is half the size, which already covers most of the original 6x6 image.

## Precision and edge cases

The sinusoidal embedding sees timesteps up to 999, so its fastest angles are large numbers whose rounding error passes straight into the cosines. When the network runs in float64, the embedding should be built in float64 as well, or float32 rounding of the large angles limits how precise the output can be. A 2x2 input is downsampled to a single pixel, and padding keeps every 3x3 convolution well defined even there.

## Where this network sits in diffusion

During training, this forward pass receives a noisy sample $x_t$ and its timestep $t$, and its output is compared with the true noise. During sampling the same forward pass runs once per reverse step, and its noise prediction drives the update from $x_t$ to $x_{t-1}$. Larger models add more resolution levels, several residual blocks per level and attention blocks at low resolution, but the wiring follows the same U shape.

## Common mistakes to avoid

- **Concatenating the skip features in the wrong order.** The up block weights expect the upsampled channels first, so swapping the halves feeds every weight the wrong input.
- **Saving the input convolution output as the skip.** The saved features must come from the down residual block, otherwise the decoder loses everything that block computed.
- **Upsampling bilinearly instead of with nearest repetition.** Bilinear interpolation produces different in-between values, so the up convolution sees a different input.
- **Dropping the SiLU inside the time MLP.** The two linear layers then act as one linear map, and every time shift changes.
- **Leaving out the final group norm and SiLU.** The output convolution then sees features of uncontrolled scale, and every predicted value changes.
- **Building the embedding in float32 for a float64 network.** Float32 rounding of the large angles then limits how precise the float64 output can be.
- **Forgetting the time shift inside the residual blocks.** The network then gives the same prediction at every noise level.
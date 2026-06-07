# <span style="font-size: 20px;">Transposed Convolution</span>

## <span style="font-size: 16px;">What is Transposed Convolution?</span>

<span style="font-size: 14px;">Transposed convolution is the mathematical adjoint of the regular convolution operation. If a convolution maps a larger spatial input to a smaller output (downsampling), the transposed convolution maps a smaller input to a larger output (upsampling).</span>

<span style="font-size: 14px;">Names used interchangeably in the literature:</span>
- <span style="font-size: 14px;">**Transposed convolution**: the mathematically precise term; it is the transpose of the convolution matrix</span>
- <span style="font-size: 14px;">**Fractional-stride convolution**: describes the implementation (inserting zeros to achieve sub-pixel stride)</span>
- <span style="font-size: 14px;">**Deconvolution**: commonly used but technically incorrect; deconvolution refers to inverting a convolution, which is a different operation</span>

<span style="font-size: 14px;">The key insight: regular convolution with stride</span> $s$ <span style="font-size: 14px;">reduces spatial dimensions by a factor of</span> $s$<span style="font-size: 14px;">. Its transpose increases spatial dimensions by a factor of</span> $s$<span style="font-size: 14px;">.</span>

## <span style="font-size: 16px;">The Transpose Relationship</span>

<span style="font-size: 14px;">Consider a regular convolution as a matrix multiplication. If we flatten the input and output into vectors, convolution is:</span>

$$
\mathbf{y} = C \, \mathbf{x}
$$

<span style="font-size: 14px;">where</span> $C$ <span style="font-size: 14px;">is a sparse Toeplitz matrix encoding the sliding-window operation. The transposed convolution uses the transpose of this matrix:</span>

$$
\mathbf{x}' = C^T \, \mathbf{y}
$$

<span style="font-size: 14px;">This is precisely what happens during backpropagation: the gradient of the loss with respect to the input of a conv layer is computed by applying the transposed convolution to the upstream gradient. Frameworks implement this as</span> `conv_transpose2d`<span style="font-size: 14px;">.</span>

<span style="font-size: 14px;">Important: the transposed convolution is NOT the inverse of convolution. It does not recover the original input; it only preserves the spatial dimensions.</span>

## <span style="font-size: 16px;">Scatter vs Gather</span>

<span style="font-size: 14px;">The most intuitive way to understand the difference:</span>
- <span style="font-size: 14px;">**Regular convolution (gather)**: for each output position, collect values from a patch of the input, multiply by the filter, and sum</span>
- <span style="font-size: 14px;">**Transposed convolution (scatter)**: for each input position, multiply by the filter and distribute (scatter) the result to a region of the output</span>

<span style="font-size: 14px;">The scatter viewpoint leads to the direct implementation: loop over input positions and add weighted contributions to the output. Multiple input positions may scatter to the same output position, and their contributions are summed.</span>

## <span style="font-size: 16px;">Fractional-Stride Implementation</span>

<span style="font-size: 14px;">An alternative implementation inserts zeros between input elements, then applies a regular convolution:</span>

- <span style="font-size: 14px;">**Step 1**: Upsample the input by inserting</span> $(s - 1)$ <span style="font-size: 14px;">zeros between each element, creating an expanded tensor of shape</span> $((H_{\text{in}} - 1) \cdot s + 1, (W_{\text{in}} - 1) \cdot s + 1)$
- <span style="font-size: 14px;">**Step 2**: Pad the expanded tensor with</span> $(k - 1 - p)$ <span style="font-size: 14px;">zeros on each side</span>
- <span style="font-size: 14px;">**Step 3**: Apply a regular convolution with the spatially-flipped filter and stride 1</span>

<span style="font-size: 14px;">This produces the same result as the scatter approach. The "fractional stride" name comes from the fact that the effective stride in the upsampled space is</span> $1/s$<span style="font-size: 14px;">.</span>

## <span style="font-size: 16px;">Output Size Formula</span>

<span style="font-size: 14px;">The output dimensions are determined by:</span>

$$
H_{\text{out}} = (H_{\text{in}} - 1) \cdot s - 2p + k_H
$$

<span style="font-size: 14px;">Notice this is the "inverse" of the regular convolution formula</span> $H_{\text{out}} = \lfloor(H + 2p - k) / s\rfloor + 1$<span style="font-size: 14px;">. If a regular convolution with stride</span> $s$ <span style="font-size: 14px;">maps</span> $H \to H'$<span style="font-size: 14px;">, the transposed convolution with the same parameters maps</span> $H' \to H$ <span style="font-size: 14px;">(recovering the original spatial size).</span>

<span style="font-size: 14px;">There is an ambiguity: different input sizes can produce the same output size after strided convolution (e.g., inputs of size 5 and 6 both produce output size 3 with kernel 3, stride 2). The</span> `output_padding` <span style="font-size: 14px;">parameter in PyTorch resolves this ambiguity by adding extra rows/columns to the output.</span>

## <span style="font-size: 16px;">The Checkerboard Artifact Problem</span>

<span style="font-size: 14px;">Transposed convolution with stride > 1 is notorious for producing **checkerboard artifacts**: a grid-like pattern of uneven magnitudes in the output. This occurs because the scatter regions overlap unevenly.</span>

<span style="font-size: 14px;">Consider stride 2 with a 3x3 kernel on a 1D slice. Input positions 0 and 1 scatter to output positions:</span>
- <span style="font-size: 14px;">Position 0: outputs 0, 1, 2</span>
- <span style="font-size: 14px;">Position 1: outputs 2, 3, 4</span>

<span style="font-size: 14px;">Output position 2 receives contributions from both inputs, while positions 0, 1, 3, 4 receive only one contribution each. This uneven overlap creates a pattern where some output pixels have roughly twice the magnitude of others.</span>

<span style="font-size: 14px;">The condition for artifact-free transposed convolution is:</span>

$$
k \mod s = 0
$$

<span style="font-size: 14px;">When the kernel size is divisible by the stride, every output position receives exactly the same number of contributions. Common artifact-free choices: $k=4, s=2$ or $k=6, s=3$.</span>

<span style="font-size: 14px;">**Alternatives that avoid checkerboard artifacts:**</span>
- <span style="font-size: 14px;">**Nearest-neighbor upsample + convolution**: upsample spatially (repeat pixels), then apply a regular convolution. Used in StyleGAN and many modern generators.</span>
- <span style="font-size: 14px;">**Bilinear upsample + convolution**: smoother interpolation before convolution. Popular in semantic segmentation (UNet).</span>
- <span style="font-size: 14px;">**Sub-pixel convolution (PixelShuffle)**: produce</span> $s^2$ <span style="font-size: 14px;">channels via regular convolution, then rearrange into spatial dimensions. Used in super-resolution (ESPCN).</span>

## <span style="font-size: 16px;">Where Transposed Convolution is Used</span>

<span style="font-size: 14px;">Transposed convolution appears in virtually every architecture that needs to upsample feature maps:</span>
- <span style="font-size: 14px;">**Autoencoders / VAEs**: the decoder upsamples the latent representation back to input resolution</span>
- <span style="font-size: 14px;">**GANs**: the generator maps a noise vector to a full-resolution image through successive upsampling layers</span>
- <span style="font-size: 14px;">**Semantic segmentation**: UNet, FCN, and DeepLab use transposed convolutions (or alternatives) to produce pixel-level predictions</span>
- <span style="font-size: 14px;">**Object detection**: Feature Pyramid Networks upsample deeper features to match shallower spatial dimensions</span>
- <span style="font-size: 14px;">**Super-resolution**: upscaling low-resolution images to high resolution</span>

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

- <span style="font-size: 14px;">**Q: Why is "deconvolution" a misnomer?**</span>
  <span style="font-size: 14px;">A: Deconvolution in signal processing means inverting a convolution (recovering the original signal), which requires solving a system of equations. Transposed convolution is the adjoint, not the inverse. Applying convolution followed by transposed convolution does not recover the original input.</span>

- <span style="font-size: 14px;">**Q: What causes checkerboard artifacts and how do you fix them?**</span>
  <span style="font-size: 14px;">A: Uneven overlap of scatter regions when $k \mod s \neq 0$. Fixes: use $k$ divisible by $s$, or replace transposed convolution with upsample + regular convolution (nearest-neighbor or bilinear interpolation followed by a 3x3 conv).</span>

- <span style="font-size: 14px;">**Q: What is the relationship between transposed convolution and the backward pass of convolution?**</span>
  <span style="font-size: 14px;">A: The gradient of a loss with respect to the input of a conv layer is exactly a transposed convolution of the upstream gradient with the same filter. This is why the operation is called "transposed": it transposes the linear mapping performed by the forward convolution.</span>

- <span style="font-size: 14px;">**Q: When would you prefer upsample + conv over transposed conv?**</span>
  <span style="font-size: 14px;">A: When avoiding checkerboard artifacts is critical (e.g., image generation). The upsample + conv approach decouples the upsampling from the learned filtering, giving smoother results. The downside is slightly more computation.</span>

- <span style="font-size: 14px;">**Q: What is PixelShuffle / sub-pixel convolution?**</span>
  <span style="font-size: 14px;">A: Instead of upsampling then convolving, produce $s^2$ output channels with a regular convolution, then rearrange the channels into spatial positions (reshape from $(C \cdot s^2, H, W)$ to $(C, H \cdot s, W \cdot s)$). This avoids checkerboard artifacts and is computationally efficient. Used in ESPCN and real-time super-resolution.</span>

---
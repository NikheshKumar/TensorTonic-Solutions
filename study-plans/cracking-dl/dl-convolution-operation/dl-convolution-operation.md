## <span style="font-size: 16px;">Convolution vs Cross-Correlation</span>

<span style="font-size: 14px;">In mathematics, convolution flips the kernel before sliding it over the input:</span>

$$
(f * g)(t) = \int f(\tau)\, g(t - \tau)\, d\tau
$$

<span style="font-size: 14px;">In deep learning, what we call "convolution" is actually **cross-correlation**: no flipping occurs. The distinction does not matter for learning because the network learns the filter weights, and a learned flipped filter is equivalent to a learned unflipped filter.</span>

<span style="font-size: 14px;">Key terminology:</span>
- <span style="font-size: 14px;">**Kernel/Filter**: the small weight matrix that slides over the input</span>
- <span style="font-size: 14px;">**Feature map**: the output of applying a filter to an input</span>
- <span style="font-size: 14px;">**Receptive field**: the region of the input that contributes to a single output element</span>

## <span style="font-size: 16px;">Multi-Channel Convolution</span>

<span style="font-size: 14px;">Real inputs have multiple channels (e.g., RGB images with 3 channels). A single filter has shape</span> $(C_{\text{in}}, k_H, k_W)$ <span style="font-size: 14px;">and produces a single 2D feature map by summing the convolution results across all input channels:</span>

$$
\text{output}(i, j) = \sum_{c=0}^{C_{\text{in}}-1} \sum_{m=0}^{k_H-1} \sum_{n=0}^{k_W-1} \text{input}(c, i+m, j+n) \cdot \text{filter}(c, m, n) + b
$$

<span style="font-size: 14px;">To produce multiple output channels, we use multiple filters. With</span> $C_{\text{out}}$ <span style="font-size: 14px;">filters, the filter tensor has shape</span> $(C_{\text{out}}, C_{\text{in}}, k_H, k_W)$ <span style="font-size: 14px;">and the output has</span> $C_{\text{out}}$ <span style="font-size: 14px;">feature maps.</span>

<span style="font-size: 14px;">This is the fundamental pattern in CNNs: each layer transforms</span> $C_{\text{in}}$ <span style="font-size: 14px;">channels into</span> $C_{\text{out}}$ <span style="font-size: 14px;">channels. Early layers detect edges and textures; deeper layers detect complex patterns.</span>

## <span style="font-size: 16px;">Padding</span>

<span style="font-size: 14px;">Without padding ("valid" mode), the output shrinks with each convolution layer:</span>

$$
H_{\text{out}} = H - k_H + 1
$$

<span style="font-size: 14px;">This causes two problems: rapid spatial dimension reduction and underrepresentation of border pixels. Zero-padding adds</span> $p$ <span style="font-size: 14px;">rows/columns of zeros around the input before convolving:</span>

$$
H_{\text{out}} = H + 2p - k_H + 1
$$

<span style="font-size: 14px;">Common padding strategies:</span>
- <span style="font-size: 14px;">**Valid** (p=0): no padding, output shrinks</span>
- <span style="font-size: 14px;">**Same** ($p = \lfloor k/2 \rfloor$ for odd kernels): output spatial size equals input spatial size (with stride 1)</span>
- <span style="font-size: 14px;">**Full** ($p = k - 1$): every input element participates in every possible position</span>

<span style="font-size: 14px;">In practice, "same" padding with odd-sized kernels (3x3, 5x5) is the most common choice because it preserves spatial resolution while maintaining a centered receptive field.</span>

## <span style="font-size: 16px;">Stride</span>

<span style="font-size: 14px;">Stride</span> $s$ <span style="font-size: 14px;">controls how many pixels the filter moves at each step. With stride</span> $s > 1$<span style="font-size: 14px;">, the output is downsampled:</span>

$$
H_{\text{out}} = \left\lfloor \frac{H + 2p - k_H}{s} \right\rfloor + 1
$$

<span style="font-size: 14px;">Strided convolutions serve the same purpose as pooling (spatial downsampling) but are learnable. Modern architectures like ResNet use strided convolutions instead of max pooling for downsampling. The advantage is that the network learns what information to preserve during downsampling rather than applying a fixed rule.</span>

## <span style="font-size: 16px;">1x1 Convolutions</span>

<span style="font-size: 14px;">A filter with</span> $k_H = k_W = 1$ <span style="font-size: 14px;">performs a linear combination across channels at each spatial location independently:</span>

$$
\text{output}(c_o, i, j) = \sum_{c=0}^{C_{\text{in}}-1} W(c_o, c) \cdot \text{input}(c, i, j) + b_{c_o}
$$

<span style="font-size: 14px;">This is equivalent to applying a fully-connected layer to each pixel's channel vector. Uses include:</span>
- <span style="font-size: 14px;">**Channel reduction**: reducing</span> $C_{\text{in}}$ <span style="font-size: 14px;">to a smaller</span> $C_{\text{out}}$ <span style="font-size: 14px;">before an expensive 3x3 or 5x5 convolution (bottleneck design in ResNet, Inception)</span>
- <span style="font-size: 14px;">**Channel expansion**: increasing feature richness without spatial mixing</span>
- <span style="font-size: 14px;">**Pointwise convolution**: the second step in depthwise separable convolutions (MobileNet)</span>

## <span style="font-size: 16px;">Parameter Count and Computational Cost</span>

<span style="font-size: 14px;">A convolutional layer with</span> $C_{\text{out}}$ <span style="font-size: 14px;">filters of shape</span> $(C_{\text{in}}, k_H, k_W)$ <span style="font-size: 14px;">has:</span>

$$
\text{Parameters} = C_{\text{out}} \times (C_{\text{in}} \times k_H \times k_W + 1)
$$

<span style="font-size: 14px;">The +1 accounts for the bias per filter. The FLOPs (multiply-accumulate operations) for producing the output are:</span>

$$
\text{FLOPs} = C_{\text{out}} \times H_{\text{out}} \times W_{\text{out}} \times C_{\text{in}} \times k_H \times k_W
$$

<span style="font-size: 14px;">Key insight for interviews: convolution is parameter-efficient because the same filter is reused (shared) across all spatial positions. A fully-connected layer connecting the same input/output dimensions would have far more parameters.</span>

## <span style="font-size: 16px;">Connection to Fully-Connected Layers</span>

<span style="font-size: 14px;">A convolution can be viewed as a sparse, weight-sharing fully-connected layer. If you unrolled the input patches into rows (im2col) and the filter into a column, convolution becomes a matrix multiplication. This is exactly how GPU implementations achieve high throughput:</span>
- <span style="font-size: 14px;">Extract all patches into an "unrolled" matrix of shape</span> $(H_{\text{out}} \times W_{\text{out}},\; C_{\text{in}} \times k_H \times k_W)$
- <span style="font-size: 14px;">Reshape filters to</span> $(C_{\text{out}},\; C_{\text{in}} \times k_H \times k_W)$
- <span style="font-size: 14px;">Output = filters @ patches.T</span>

<span style="font-size: 14px;">This "im2col" trick trades memory for speed and is the standard approach in cuDNN.</span>

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

- <span style="font-size: 14px;">**Q: What is the difference between convolution and cross-correlation in the context of CNNs?**</span>
  <span style="font-size: 14px;">A: Mathematically, convolution flips the kernel, cross-correlation does not. In practice, CNNs learn the kernel weights, so flipping is irrelevant. All DL frameworks implement cross-correlation but call it "convolution."</span>

- <span style="font-size: 14px;">**Q: Why do we use small kernels (3x3) instead of large ones (7x7)?**</span>
  <span style="font-size: 14px;">A: Two stacked 3x3 convolutions have the same receptive field as one 5x5 but with fewer parameters (2 * 9 = 18 vs 25) and an extra non-linearity between them. VGG demonstrated this principle. Three 3x3 layers match a 7x7 receptive field with 27 vs 49 parameters per channel.</span>

- <span style="font-size: 14px;">**Q: How does "same" padding interact with even-sized kernels?**</span>
  <span style="font-size: 14px;">A: For even kernel sizes, symmetric padding cannot perfectly preserve dimensions. This is why odd kernels (3x3, 5x5, 7x7) are strongly preferred: they have a well-defined center pixel and symmetric padding.</span>

- <span style="font-size: 14px;">**Q: What is a dilated (atrous) convolution?**</span>
  <span style="font-size: 14px;">A: Dilated convolutions insert gaps (dilation factor - 1 zeros) between kernel elements, expanding the receptive field without increasing parameters. A 3x3 kernel with dilation 2 covers a 5x5 area using only 9 weights. Used in semantic segmentation (DeepLab) and WaveNet.</span>

- <span style="font-size: 14px;">**Q: Can you explain depthwise separable convolutions?**</span>
  <span style="font-size: 14px;">A: Split convolution into two steps: (1) depthwise: one filter per input channel (no cross-channel mixing), (2) pointwise: 1x1 convolution to mix channels. This reduces parameters from</span> $C_{\text{out}} \cdot C_{\text{in}} \cdot k^2$ <span style="font-size: 14px;">to</span> $C_{\text{in}} \cdot k^2 + C_{\text{in}} \cdot C_{\text{out}}$<span style="font-size: 14px;">. MobileNet and EfficientNet are built entirely on this idea.</span>

- <span style="font-size: 14px;">**Q: How does the backward pass of convolution work?**</span>
  <span style="font-size: 14px;">A: The gradient with respect to the input is a "full" convolution of the upstream gradient with the flipped filter. The gradient with respect to the filter is a valid cross-correlation of the input with the upstream gradient. This duality between forward and backward is a key reason convolution works elegantly in neural networks.</span>

---
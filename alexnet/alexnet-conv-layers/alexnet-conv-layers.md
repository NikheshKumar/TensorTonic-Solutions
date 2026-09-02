# <span style="font-size: 20px;">AlexNet Convolution Layers</span>

<span style="font-size: 14px;">A **convolutional layer** applies learnable filters to an input tensor, producing feature maps that detect local spatial patterns (edges, textures, shapes). Krizhevsky et al. (2012) used five convolutional layers as the backbone of AlexNet, the model that won ImageNet LSVRC-2012 and reignited modern deep learning.</span>

---

## <span style="font-size: 16px;">What a Convolution Layer Does</span>

<span style="font-size: 14px;">A convolutional layer slides a small filter (**kernel**) across an input tensor. At every position, the filter computes a dot product with the input patch it covers, producing a single scalar. Repeating this across all valid positions generates a 2D **feature map** (or **activation map**).</span>

* <span style="font-size: 14px;">**Multiple filters:** Each filter detects one type of pattern. With $F$ filters, the output has $F$ channels.</span>
* <span style="font-size: 14px;">**Multi-channel input:** Each filter spans all input channels. A filter applied to $C_{in}$ input channels has shape $k \times k \times C_{in}$.</span>
* <span style="font-size: 14px;">**Parameters:** The layer has $F$ filters plus $F$ bias terms (one per filter).</span>

---

## <span style="font-size: 16px;">The Output Dimension Formula</span>

<span style="font-size: 14px;">For each spatial axis:</span>

$$
H_{out} = \lfloor \frac{H_{in} + 2p - k}{s} \rfloor + 1
$$

* <span style="font-size: 14px;">$H_{in}$: input spatial dimension</span>
* <span style="font-size: 14px;">$k$: kernel size</span>
* <span style="font-size: 14px;">$s$: stride (pixels the kernel moves between applications)</span>
* <span style="font-size: 14px;">$p$: padding (zero-pixels added to each side before convolution)</span>

<span style="font-size: 14px;">After padding, effective input size is $H_{in} + 2p$. The kernel's furthest valid start position is $H_{in} + 2p - k$. Dividing by stride gives the number of steps; adding 1 accounts for the initial position. The floor discards any leftover pixels that don't fit a complete kernel window.</span>

<span style="font-size: 14px;">Full output tensor shape:</span>

$$
(B, H_{out}, W_{out}, F)
$$

<span style="font-size: 14px;">where $B$ is the batch size. The width formula is identical with $W_{in}$ substituted for $H_{in}$.</span>

---

## <span style="font-size: 16px;">What Each Parameter Controls</span>

### <span style="font-size: 14px;">Kernel Size</span>

<span style="font-size: 14px;">Kernel size $k$ defines the local neighborhood each filter sees. Larger kernels have a larger **receptive field** but more parameters ($k^2 \times C_{in}$ per filter). AlexNet uses $11 \times 11$ in Conv1 (to capture large-scale edges and color gradients from raw pixels) and progressively smaller kernels ($5 \times 5$, then $3 \times 3$) in deeper layers where each position already represents a large image region.</span>

### <span style="font-size: 14px;">Stride</span>

<span style="font-size: 14px;">Stride $s$ controls how far the kernel moves between applications. Stride 1 produces maximum output size; stride 4 reduces output size by roughly 4x per spatial dimension. Unlike pooling, strided convolution downsamples while learning features. AlexNet's stride 4 in Conv1 aggressively reduces $224 \times 224$ to $55 \times 55$ in one layer -- a pragmatic choice for the GPUs available in 2012.</span>

### <span style="font-size: 14px;">Padding</span>

<span style="font-size: 14px;">Padding adds zero-rows and zero-columns around the input border. Without it, each layer shrinks spatial dimensions by $k - 1$ pixels (at stride 1).</span>

* <span style="font-size: 14px;">**Valid padding** ($p = 0$): no padding, output smaller than input.</span>
* <span style="font-size: 14px;">**Same padding**: $p$ chosen so output size equals $\lceil H_{in} / s \rceil$, preserving dimensions at stride 1.</span>

<span style="font-size: 14px;">AlexNet uses specific per-layer values: padding 2 for Conv1 and Conv2, padding 1 for Conv3-5.</span>

### <span style="font-size: 14px;">Number of Filters</span>

<span style="font-size: 14px;">The number of filters $F$ determines feature diversity. More filters = more capacity, more parameters. AlexNet uses 96 -> 256 -> 384 -> 384 -> 256 filters across Conv1-5. This pattern of increasing channels while decreasing spatial size is a hallmark of CNN architectures: as resolution drops, the network compensates with more feature channels.</span>

---

## <span style="font-size: 16px;">AlexNet's Five Convolutional Layers</span>

<span style="font-size: 14px;">AlexNet has five convolutional layers followed by three fully connected layers. The complete convolutional stack:</span>

* <span style="font-size: 14px;">**Conv1**: 96 filters, kernel $11 \times 11$, stride 4, padding 2. Input $(224, 224, 3)$, output $(55, 55, 96)$. Followed by ReLU, LRN, max pooling ($3 \times 3$, stride 2) -> $(27, 27, 96)$.</span>
* <span style="font-size: 14px;">**Conv2**: 256 filters, kernel $5 \times 5$, stride 1, padding 2. Input $(27, 27, 96)$, output $(27, 27, 256)$. Followed by ReLU, LRN, max pooling ($3 \times 3$, stride 2) -> $(13, 13, 256)$.</span>
* <span style="font-size: 14px;">**Conv3**: 384 filters, kernel $3 \times 3$, stride 1, padding 1. Input $(13, 13, 256)$, output $(13, 13, 384)$. ReLU only (no LRN, no pooling).</span>
* <span style="font-size: 14px;">**Conv4**: 384 filters, kernel $3 \times 3$, stride 1, padding 1. Input $(13, 13, 384)$, output $(13, 13, 384)$. ReLU only.</span>
* <span style="font-size: 14px;">**Conv5**: 256 filters, kernel $3 \times 3$, stride 1, padding 1. Input $(13, 13, 384)$, output $(13, 13, 256)$. ReLU + max pooling ($3 \times 3$, stride 2) -> $(6, 6, 256)$.</span>

<span style="font-size: 14px;">After Conv5 pooling, the output is flattened to $6 \times 6 \times 256 = 9{,}216$ elements feeding the fully connected layers. The conv layers contain most of the computation but a minority of the parameters; the FC layers contain the majority of parameters.</span>

---

## <span style="font-size: 16px;">The Paper's Design Decisions</span>

### <span style="font-size: 14px;">Why $11 \times 11$ Kernels in the First Layer</span>

<span style="font-size: 14px;">Conv1 operates on raw RGB pixels. An $11 \times 11$ kernel covers 121 pixels per channel -- large enough to detect oriented edges, color blobs, and texture gradients. Smaller kernels would have too narrow a view at this stage. Combined with stride 4, adjacent output positions have overlapping receptive fields ($11 - 4 = 7$ pixel overlap), downsampling from $224 \times 224$ to $55 \times 55$ in one layer.</span>

### <span style="font-size: 14px;">Why Smaller Kernels in Deeper Layers</span>

<span style="font-size: 14px;">By Conv3-5, each spatial position's **effective receptive field** already covers a large image region (each pixel summarizes many original pixels from prior layers). A $3 \times 3$ kernel at this depth is sufficient to combine high-level features.</span>

### <span style="font-size: 14px;">GPU Memory Constraints in 2012</span>

<span style="font-size: 14px;">AlexNet was split across two GTX 580 GPUs (3 GB each). Each GPU handled half the filters in most layers. Conv3 is an exception: it receives all 256 channels from both GPUs, while Conv4 and Conv5 connect only within the same GPU. This hardware-specific optimization influenced the filter counts (96, 256, 384, 384, 256) but is no longer relevant with modern GPUs.</span>

### <span style="font-size: 14px;">The Role of LRN and Pooling Between Layers</span>

<span style="font-size: 14px;">**Local Response Normalization (LRN)** after Conv1 and Conv2 normalizes neuron responses by dividing by summed squared responses of neighbors across adjacent feature maps. The paper reports LRN reduces top-1 error by 1.4% and top-5 by 1.2%. Later architectures (VGGNet, ResNet) found LRN unnecessary and replaced it with Batch Normalization.</span>

<span style="font-size: 14px;">**Max pooling** ($3 \times 3$, stride 2) after Conv1, Conv2, and Conv5 uses **overlapping pooling** (pool 3, stride 2). Krizhevsky et al. report this reduces top-1 error by 0.4% and top-5 by 0.3% vs. non-overlapping pooling.</span>

---

## <span style="font-size: 16px;">Parameter Count</span>

<span style="font-size: 14px;">Parameters per convolutional layer:</span>

$$
\text{params} = F \times (k^2 \times C_{in} + 1)
$$

<span style="font-size: 14px;">The $+1$ accounts for one bias per filter. For AlexNet:</span>

* <span style="font-size: 14px;">**Conv1**: $96 \times (11^2 \times 3 + 1) = 96 \times 364 = 34{,}944$</span>
* <span style="font-size: 14px;">**Conv2**: $256 \times (5^2 \times 96 + 1) = 256 \times 2{,}401 = 614{,}656$</span>
* <span style="font-size: 14px;">**Conv3**: $384 \times (3^2 \times 256 + 1) = 384 \times 2{,}305 = 885{,}120$</span>
* <span style="font-size: 14px;">**Conv4**: $384 \times (3^2 \times 384 + 1) = 384 \times 3{,}457 = 1{,}327{,}488$</span>
* <span style="font-size: 14px;">**Conv5**: $256 \times (3^2 \times 384 + 1) = 256 \times 3{,}457 = 884{,}992$</span>

<span style="font-size: 14px;">Total conv parameters: ~$3.7$ million. By comparison, the first FC layer alone has $9{,}216 \times 4{,}096 = 37{,}748{,}736$ parameters (roughly 10x all conv layers combined). This illustrates that conv layers are parameter-efficient due to weight sharing across spatial positions.</span>

---

## <span style="font-size: 16px;">Computational Cost (FLOPs)</span>

<span style="font-size: 14px;">FLOPs per convolutional layer:</span>

$$
\text{FLOPs} \approx 2 \times H_{out} \times W_{out} \times F \times k^2 \times C_{in}
$$

<span style="font-size: 14px;">The factor of 2 accounts for multiply-accumulate operations. For Conv1:</span>

$$
\text{FLOPs}_{\text{Conv1}} \approx 2 \times 55 \times 55 \times 96 \times 11^2 \times 3 \approx 211 \text{ million}
$$

<span style="font-size: 14px;">Conv layers dominate computation (~90% of total FLOPs) while FC layers dominate parameter count. This asymmetry is characteristic of CNNs.</span>

---

## <span style="font-size: 16px;">Numerical Example: Conv1 Shape Computation</span>

<span style="font-size: 14px;">Input: batch of 2 RGB images, each $224 \times 224$. Tensor shape: $(2, 224, 224, 3)$. Conv1: $k = 11$, $s = 4$, $p = 2$, $F = 96$.</span>

<span style="font-size: 14px;">1. **After padding**: $224 + 2 \times 2 = 228$ per axis.</span>

<span style="font-size: 14px;">2. **Valid kernel placements**: Span available: $228 - 11 = 217$. Steps: $\lfloor 217 / 4 \rfloor = 54$. Plus starting position: $54 + 1 = 55$.</span>

<span style="font-size: 14px;">3. **Formula verification**: $H_{out} = \lfloor \frac{224 + 4 - 11}{4} \rfloor + 1 = \lfloor \frac{217}{4} \rfloor + 1 = 54 + 1 = 55$.</span>

<span style="font-size: 14px;">4. **Width**: Same (square input). $W_{out} = 55$.</span>

<span style="font-size: 14px;">5. **Output channels**: $C_{out} = 96$.</span>

<span style="font-size: 14px;">6. **Final output shape**: $(2, 55, 55, 96)$.</span>

<span style="font-size: 14px;">7. **Parameters**: $96 \times (11 \times 11 \times 3 + 1) = 96 \times 364 = 34{,}944$.</span>

---

## <span style="font-size: 16px;">Numerical Example: Tracing Through All Five Layers</span>

<span style="font-size: 14px;">Starting from $(1, 224, 224, 3)$:</span>

### <span style="font-size: 14px;">Conv1</span>

<span style="font-size: 14px;">$k = 11$, $s = 4$, $p = 2$, $F = 96$.</span>

$$
H_{out} = \lfloor \frac{224 + 4 - 11}{4} \rfloor + 1 = 55
$$

<span style="font-size: 14px;">Output: $(1, 55, 55, 96)$. After ReLU, LRN, max pooling ($k_p = 3$, $s_p = 2$):</span>

$$
H_{pool} = \lfloor \frac{55 - 3}{2} \rfloor + 1 = 27
$$

<span style="font-size: 14px;">After pooling: $(1, 27, 27, 96)$.</span>

### <span style="font-size: 14px;">Conv2</span>

<span style="font-size: 14px;">$k = 5$, $s = 1$, $p = 2$, $F = 256$.</span>

$$
H_{out} = \lfloor \frac{27 + 4 - 5}{1} \rfloor + 1 = 27
$$

<span style="font-size: 14px;">Output: $(1, 27, 27, 256)$. After ReLU, LRN, max pooling ($k_p = 3$, $s_p = 2$):</span>

$$
H_{pool} = \lfloor \frac{27 - 3}{2} \rfloor + 1 = 13
$$

<span style="font-size: 14px;">After pooling: $(1, 13, 13, 256)$.</span>

### <span style="font-size: 14px;">Conv3</span>

<span style="font-size: 14px;">$k = 3$, $s = 1$, $p = 1$, $F = 384$.</span>

$$
H_{out} = \lfloor \frac{13 + 2 - 3}{1} \rfloor + 1 = 13
$$

<span style="font-size: 14px;">Output: $(1, 13, 13, 384)$. ReLU only.</span>

### <span style="font-size: 14px;">Conv4</span>

<span style="font-size: 14px;">$k = 3$, $s = 1$, $p = 1$, $F = 384$.</span>

$$
H_{out} = \lfloor \frac{13 + 2 - 3}{1} \rfloor + 1 = 13
$$

<span style="font-size: 14px;">Output: $(1, 13, 13, 384)$. ReLU only.</span>

### <span style="font-size: 14px;">Conv5</span>

<span style="font-size: 14px;">$k = 3$, $s = 1$, $p = 1$, $F = 256$.</span>

$$
H_{out} = \lfloor \frac{13 + 2 - 3}{1} \rfloor + 1 = 13
$$

<span style="font-size: 14px;">Output: $(1, 13, 13, 256)$. After ReLU, max pooling ($k_p = 3$, $s_p = 2$):</span>

$$
H_{pool} = \lfloor \frac{13 - 3}{2} \rfloor + 1 = 6
$$

<span style="font-size: 14px;">After pooling: $(1, 6, 6, 256)$. Flattened to $(1, 9216)$ before FC layers.</span>

<span style="font-size: 14px;">Spatial progression: $224 \to 55 \to 27 \to 27 \to 13 \to 13 \to 13 \to 13 \to 6$. Aggressive downsampling happens in Conv1 (stride 4) and the three pooling stages. Conv3 and Conv4 use same-padding to preserve spatial size.</span>

---

## <span style="font-size: 16px;">From AlexNet to Modern Convolutions</span>

### <span style="font-size: 14px;">VGGNet: Only $3 \times 3$ Kernels</span>

<span style="font-size: 14px;">Simonyan and Zisserman (2014) showed that stacking $3 \times 3$ layers matches large-kernel receptive fields with fewer parameters and more non-linearity. Two $3 \times 3$ layers = $5 \times 5$ receptive field with $2 \times 9 = 18$ weights vs. $25$. Three $3 \times 3$ layers = $11 \times 11$ field with $27$ weights vs. $121$.</span>

### <span style="font-size: 14px;">ResNet: Residual Connections</span>

<span style="font-size: 14px;">He et al. (2015) introduced skip connections ($\text{output} = F(x) + x$), enabling training of 50-152 layer networks by addressing vanishing gradients. ResNet uses $1 \times 1$ convolutions for channel adjustment and a $7 \times 7$ first-layer kernel at stride 2 followed by $3 \times 3$ max pool.</span>

### <span style="font-size: 14px;">Depthwise Separable Convolutions</span>

<span style="font-size: 14px;">MobileNet (Howard et al., 2017) decomposed standard convolution into depthwise ($k \times k \times 1$ per channel) and pointwise ($1 \times 1 \times C_{in}$) convolutions. For AlexNet's Conv3: standard = $384 \times 3^2 \times 256 = 884{,}736$ MACs per output position; depthwise separable = $3^2 \times 256 + 256 \times 384 = 100{,}608$ (8.8x reduction).</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Off-by-one in the output formula.** Forgetting the floor or the $+1$ in $\lfloor(H_{in} + 2p - k)/s\rfloor + 1$ produces incorrect shapes that crash subsequent layers with dimension mismatch errors.</span>
* <span style="font-size: 14px;">**Forgetting padding is added to both sides.** The formula uses $2p$, not $p$. Padding 2 means 2 pixels on each side (4 extra per axis). Using $p$ instead of $2p$ produces an output that is too small.</span>
* <span style="font-size: 14px;">**Confusing filter count with parameter count.** 96 filters does not mean 96 parameters. Each filter has $k^2 \times C_{in} + 1$ parameters. Conv1 has 96 filters but $34{,}944$ parameters.</span>
* <span style="font-size: 14px;">**Ignoring pooling between conv layers.** Conv1 outputs $(55, 55, 96)$ but Conv2 receives $(27, 27, 96)$ due to max pooling. Skipping the pooling step gives wrong input shapes for the next layer.</span>
* <span style="font-size: 14px;">**Assuming same-padding always preserves size.** Same-padding preserves size only at stride 1. With stride $s > 1$, output is $\lceil H_{in} / s \rceil$, which is smaller. No padding preserves $224 \times 224$ at stride 4.</span>
* <span style="font-size: 14px;">**Confusing stride with dilation.** Stride = how far the kernel moves. Dilation = spacing between kernel elements. AlexNet uses no dilation, but modern architectures (DeepLab) do. A dilated $3 \times 3$ kernel with rate 2 has effective size $2(3-1)+1 = 5$, changing the formula to use $k_{\text{eff}}$.</span>
* <span style="font-size: 14px;">**Applying width formula with height values on non-square inputs.** For AlexNet's square $224 \times 224$ input, $H_{out} = W_{out}$. For non-square inputs, each axis must be computed independently.</span>

---
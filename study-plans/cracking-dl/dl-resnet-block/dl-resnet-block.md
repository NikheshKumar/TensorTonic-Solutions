# <span style="font-size: 20px;">ResNet Block (Skip Connections)</span>

## <span style="font-size: 16px;">The Degradation Problem</span>

<span style="font-size: 14px;">Before ResNet, a surprising observation blocked progress in deep learning: adding more layers to a plain network (without skip connections) caused higher training error, not just higher test error. This was not overfitting - a 56-layer plain network had worse training accuracy than a 20-layer one. The optimization landscape of very deep networks created difficulties that standard SGD could not overcome.</span>

<span style="font-size: 14px;">He et al. (2015) reasoned that if the added layers could learn the identity function, the deeper network should be at least as good as the shallower one. The fact that it performed worse meant the network was struggling to learn identity mappings through sequences of non-linear layers.</span>

## <span style="font-size: 16px;">Residual Learning Framework</span>

<span style="font-size: 14px;">The key insight: instead of learning $\mathcal{H}(x)$ directly, let the network learn the residual $\mathcal{F}(x) = \mathcal{H}(x) - x$, so the output becomes:</span>

$$
\mathcal{H}(x) = \mathcal{F}(x) + x
$$

<span style="font-size: 14px;">If the optimal transformation is close to identity, $\mathcal{F}(x)$ should be close to zero, which is easier to learn than the full mapping. The skip connection provides a "gradient highway" during backpropagation:</span>

$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial x} &= \frac{\partial \mathcal{L}}{\partial \mathcal{H}} \cdot \bigl(1 + \frac{\partial \mathcal{F}}{\partial x}\bigr)
\end{aligned}
$$

<span style="font-size: 14px;">The "1" term means gradients flow directly through the skip connection, even if $\frac{\partial \mathcal{F}}{\partial x}$ is small. This prevents the vanishing gradient problem that plagued very deep plain networks.</span>

## <span style="font-size: 16px;">Block Variants</span>

<span style="font-size: 14px;">ResNet uses two types of residual blocks:</span>

<span style="font-size: 14px;">**Basic Block** (ResNet-18, ResNet-34):</span>
- <span style="font-size: 14px;">Two 3x3 convolutions, each followed by BatchNorm</span>
- <span style="font-size: 14px;">ReLU after first BN and after the addition</span>
- <span style="font-size: 14px;">This exercise uses a simplified basic block: two biased convolutions and an identity skip, with BatchNorm omitted. Use the supplied weights and biases, NumPy float64 arithmetic, and the operation order in the description.</span>

<span style="font-size: 14px;">**Bottleneck Block** (ResNet-50, ResNet-101, ResNet-152):</span>
- <span style="font-size: 14px;">1x1 conv (reduce channels) -> 3x3 conv -> 1x1 conv (expand channels)</span>
- <span style="font-size: 14px;">More parameter-efficient for deeper networks (reduces the 3x3 conv's channel count)</span>
- <span style="font-size: 14px;">Example: 256 -> 64 (1x1) -> 64 (3x3) -> 256 (1x1). The 3x3 conv operates on only 64 channels instead of 256.</span>

## <span style="font-size: 16px;">Shortcut Connection Types</span>

<span style="font-size: 14px;">The original paper explored three options:</span>

- <span style="font-size: 14px;">**Option A (zero-padding):** pad the identity with zeros to match increased channels. No extra parameters, but underperforms.</span>
- <span style="font-size: 14px;">**Option B (projection shortcuts only for dimension changes):** use 1x1 conv+BN when channels or spatial dims change, identity otherwise. This is the standard approach used in practice.</span>
- <span style="font-size: 14px;">**Option C (projection everywhere):** use 1x1 conv for every shortcut. Slightly more parameters with minimal accuracy gain over Option B.</span>

<span style="font-size: 14px;">Option B is the de facto standard. The projection shortcut adds $C_{\text{in}} \times C_{\text{out}}$ parameters (1x1 conv, no bias) plus $2 C_{\text{out}}$ BN parameters.</span>

## <span style="font-size: 16px;">Pre-activation vs Post-activation</span>

<span style="font-size: 14px;">The original ResNet uses post-activation residual blocks with BatchNorm:</span>

$$
y = \text{ReLU}(\text{BN}(\text{Conv}(\text{ReLU}(\text{BN}(\text{Conv}(x))))) + \text{shortcut}(x))
$$

<span style="font-size: 14px;">For this exercise, omit BatchNorm and use the unchanged input as the skip. The required computation is:</span>

$$
A_1=\operatorname{ReLU}(\operatorname{Conv}(X,W_1,b_1))
$$

$$
Z_2=\operatorname{Conv}(A_1,W_2,b_2)
$$

$$
Y=\operatorname{ReLU}(Z_2+X)
$$

<span style="font-size: 14px;">X is the channel-first input. W1 and W2 are the supplied 3 by 3 filters; b1 and b2 are their supplied channel biases. Each convolution is cross-correlation with stride one and one cell of zero padding. Return conv1 as A1, conv2 as Z2, and output as Y in a dictionary of nested lists. The second convolution remains unactivated until after the skip is added. The following architecture discussion describes general ResNet variants.</span>

<span style="font-size: 14px;">He et al. (2016) later proposed pre-activation blocks (ResNet-v2):</span>

$$
y = \text{Conv}(\text{ReLU}(\text{BN}(\text{Conv}(\text{ReLU}(\text{BN}(x)))))) + x
$$

<span style="font-size: 14px;">Pre-activation moves BN and ReLU before each conv. This creates a cleaner identity path (the skip connection is a pure addition without going through BN or ReLU), which empirically improves optimization for very deep networks (1000+ layers). For typical depths (50-152 layers), both variants perform similarly.</span>

## <span style="font-size: 16px;">ResNet Architecture Overview</span>

<span style="font-size: 14px;">A full ResNet stacks multiple residual blocks in stages, doubling channels and halving spatial dimensions at each stage transition:</span>

<span style="font-size: 14px;">**ResNet-18:** [2, 2, 2, 2] basic blocks, channels [64, 128, 256, 512]</span>
<span style="font-size: 14px;">**ResNet-34:** [3, 4, 6, 3] basic blocks, channels [64, 128, 256, 512]</span>
<span style="font-size: 14px;">**ResNet-50:** [3, 4, 6, 3] bottleneck blocks, channels [256, 512, 1024, 2048]</span>
<span style="font-size: 14px;">**ResNet-101:** [3, 4, 23, 3] bottleneck blocks</span>
<span style="font-size: 14px;">**ResNet-152:** [3, 8, 36, 3] bottleneck blocks</span>

<span style="font-size: 14px;">The first block of each stage (except the first) uses stride=2 to downsample. Within a stage, all other blocks use stride=1 and identity shortcuts. Global average pooling replaces the large FC layers of VGG, dramatically reducing parameters.</span>

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

- <span style="font-size: 14px;">**Q: Why does the skip connection solve the degradation problem?**</span>
  <span style="font-size: 14px;">A: The skip connection creates a gradient highway. During backpropagation, the gradient through the skip is exactly 1 (identity), so even if the conv path has very small gradients, the total gradient is at least 1 plus the conv path gradient. This prevents gradients from vanishing through many layers.</span>

- <span style="font-size: 14px;">**Q: When is a projection shortcut needed?**</span>
  <span style="font-size: 14px;">A: When the input and output dimensions do not match for element-wise addition. This happens when (1) the number of channels changes ($C_{\text{in}} \neq C_{\text{out}}$), or (2) spatial dimensions are reduced (stride > 1). The 1x1 convolution with appropriate stride handles both simultaneously.</span>

- <span style="font-size: 14px;">**Q: What is the difference between ResNet-v1 and ResNet-v2?**</span>
  <span style="font-size: 14px;">A: v1 uses post-activation (BN and ReLU after conv, ReLU after addition). v2 uses pre-activation (BN and ReLU before conv), creating a cleaner identity path. v2 is theoretically better for very deep networks but in practice the difference is small for typical depths.</span>

- <span style="font-size: 14px;">**Q: Why use bias=False in convolutions followed by BatchNorm?**</span>
  <span style="font-size: 14px;">A: BatchNorm first subtracts the batch mean, which eliminates any constant offset from the conv bias. Then BN adds its own learnable bias (beta). So the conv bias is redundant - it gets subtracted out and replaced by beta. Removing it saves a small number of parameters.</span>

- <span style="font-size: 14px;">**Q: How does ResNet compare to DenseNet?**</span>
  <span style="font-size: 14px;">A: ResNet adds the skip connection ($\mathcal{F}(x) + x$), while DenseNet concatenates ($[\mathcal{F}(x); x]$). DenseNet preserves all previous feature maps, enabling feature reuse and reducing the number of parameters needed. However, DenseNet's memory footprint is higher due to concatenation.</span>

---
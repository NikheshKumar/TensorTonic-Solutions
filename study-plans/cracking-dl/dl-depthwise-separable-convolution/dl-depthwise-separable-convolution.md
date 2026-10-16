# <span style="font-size: 20px;">Depthwise Separable Convolution</span>

## <span style="font-size: 16px;">The Efficiency Problem</span>

<span style="font-size: 14px;">Standard convolutions are computationally expensive because every output channel depends on every input channel at every spatial position. For a Conv2d mapping $C_{\text{in}}$ to $C_{\text{out}}$ with kernel size $k$, the computational cost per spatial position is $C_{\text{in}} \times C_{\text{out}} \times k^2$ multiply-accumulate operations.</span>

<span style="font-size: 14px;">For a typical layer in VGG (256 to 256 channels, 3x3 kernel), each output pixel requires $256 \times 256 \times 9 = 589{,}824$ operations. For a 14x14 feature map, that is 115 million operations per layer. This makes standard convolutions impractical for mobile and embedded devices.</span>

<span style="font-size: 14px;">Depthwise separable convolutions solve this by factorizing the standard convolution into two cheaper operations.</span>

## <span style="font-size: 16px;">Depthwise Convolution</span>

<span style="font-size: 14px;">A depthwise convolution applies a single filter per input channel. Instead of a 4D weight tensor of shape $(C_{\text{out}}, C_{\text{in}}, k, k)$, the depthwise conv has shape $(C_{\text{in}}, 1, k, k)$: one $k \times k$ filter per channel.</span>

<span style="font-size: 14px;">The operation for each channel $c$:</span>

$$
\text{out}(c, i, j) = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} \text{input}(c, i+m, j+n) \cdot W(c, m, n)
$$

<span style="font-size: 14px;">Key properties:</span>
- <span style="font-size: 14px;">No cross-channel interaction: each channel is filtered independently</span>
- <span style="font-size: 14px;">Parameters: $C_{\text{in}} \times k^2$ (no bias)</span>
- <span style="font-size: 14px;">FLOPs per spatial position: $C_{\text{in}} \times k^2$</span>
- <span style="font-size: 14px;">In PyTorch: `nn.Conv2d(C, C, k, groups=C)`</span>

<span style="font-size: 14px;">The groups parameter splits both input and output channels into groups, with each group processed independently. Setting groups=$C_{\text{in}}$ makes each group have exactly 1 input channel and 1 output channel.</span>

## <span style="font-size: 16px;">Pointwise Convolution</span>

<span style="font-size: 14px;">The pointwise convolution is a standard 1x1 convolution that mixes channels without any spatial processing:</span>

$$
\text{out}(c_{\text{out}}, i, j) = \sum_{c=0}^{C_{\text{in}}-1} \text{input}(c, i, j) \cdot W(c_{\text{out}}, c)
$$

<span style="font-size: 14px;">Key properties:</span>
- <span style="font-size: 14px;">Full cross-channel interaction: every output channel depends on all input channels</span>
- <span style="font-size: 14px;">Parameters: $C_{\text{in}} \times C_{\text{out}}$ (no bias)</span>
- <span style="font-size: 14px;">Acts as a per-pixel fully-connected layer</span>
- <span style="font-size: 14px;">Handles the channel dimension change from $C_{\text{in}}$ to $C_{\text{out}}$</span>

## <span style="font-size: 16px;">Parameter and FLOP Reduction</span>

<span style="font-size: 14px;">The reduction from depthwise separable factorization is substantial:</span>

$$
\begin{aligned}
\frac{\text{Separable params}}{\text{Standard params}} &= \frac{C_{\text{in}} \times k^2 + C_{\text{in}} \times C_{\text{out}}}{C_{\text{in}} \times C_{\text{out}} \times k^2} \\
&= \frac{1}{C_{\text{out}}} + \frac{1}{k^2}
\end{aligned}
$$

<span style="font-size: 14px;">For typical values ($C_{\text{out}} = 64$, $k = 3$):</span>

$$
\frac{1}{64} + \frac{1}{9} \approx 0.127 \quad \text{(8x reduction)}
$$

<span style="font-size: 14px;">Concrete example with $C_{\text{in}} = 32$, $C_{\text{out}} = 64$, $k = 3$:</span>

$$
\begin{aligned}
\text{Standard:} & \quad 32 \times 64 \times 3 \times 3 = 18{,}432 \text{ params} \\
\text{Depthwise:} & \quad 32 \times 3 \times 3 = 288 \text{ params} \\
\text{Pointwise:} & \quad 32 \times 64 = 2{,}048 \text{ params} \\
\text{Separable total:} & \quad 288 + 2{,}048 = 2{,}336 \text{ params (7.9x smaller)}
\end{aligned}
$$

<span style="font-size: 14px;">When including BatchNorm parameters (2 per channel per BN layer), the total becomes 2,336 + 64 + 128 = 2,528 for the separable version vs 18,432 + 128 = 18,560 for standard conv+BN.</span>

## <span style="font-size: 16px;">MobileNet Architecture Family</span>

<span style="font-size: 14px;">Depthwise separable convolutions are the core building block of the MobileNet family:</span>

<span style="font-size: 14px;">**MobileNet v1 (2017):** Replaces every standard conv in a VGG-like network with depthwise separable convs. Introduces width multiplier ($\alpha$) and resolution multiplier ($\rho$) to scale the network for different computational budgets. Achieves similar accuracy to VGG with 30x fewer parameters.</span>

<span style="font-size: 14px;">**MobileNet v2 (2018):** Introduces the inverted residual block (also called linear bottleneck). Instead of reduce-convolve-expand (like ResNet bottleneck), MobileNet v2 does expand-depthwise-reduce:</span>
- <span style="font-size: 14px;">1x1 pointwise expansion (channels multiplied by expansion factor, typically 6)</span>
- <span style="font-size: 14px;">3x3 depthwise convolution (in the expanded space)</span>
- <span style="font-size: 14px;">1x1 pointwise projection (compress back to narrow channels)</span>
- <span style="font-size: 14px;">Skip connection on the narrow (not expanded) representation</span>

<span style="font-size: 14px;">**MobileNet v3 (2019):** Adds squeeze-and-excitation attention, h-swish activation, and uses neural architecture search (NAS) to find optimal block configurations.</span>

## <span style="font-size: 16px;">Grouped Convolutions: The General Case</span>

<span style="font-size: 14px;">Depthwise convolution is the extreme case of grouped convolution where groups = $C_{\text{in}}$. The general grouped convolution with $G$ groups:</span>

- <span style="font-size: 14px;">Splits input into $G$ groups of $C_{\text{in}}/G$ channels each</span>
- <span style="font-size: 14px;">Applies independent convolutions per group</span>
- <span style="font-size: 14px;">Concatenates results</span>
- <span style="font-size: 14px;">Parameters: $C_{\text{in}} \times C_{\text{out}} \times k^2 / G$</span>

<span style="font-size: 14px;">AlexNet (2012) used groups=2 to split the network across two GPUs. ResNeXt (2017) uses groups=32 as a design principle ("cardinality"). ShuffleNet adds a channel shuffle operation between grouped convs to enable cross-group information flow.</span>

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

- <span style="font-size: 14px;">**Q: Why can we factorize spatial and channel operations without losing accuracy?**</span>
  <span style="font-size: 14px;">A: Empirically, the cross-channel and spatial correlations in feature maps are largely independent. The depthwise conv captures spatial patterns per channel, and the pointwise conv learns channel combinations. The factorization loses some representational capacity (the set of functions it can represent is a subset of standard conv), but in practice the difference is small because the representations learned by deep networks are approximately low-rank.</span>

- <span style="font-size: 14px;">**Q: What is the computational cost reduction in terms of FLOPs?**</span>
  <span style="font-size: 14px;">A: For a feature map of size $H \times W$, standard conv costs $H \times W \times C_{\text{in}} \times C_{\text{out}} \times k^2$ FLOPs. Separable conv costs $H \times W \times (C_{\text{in}} \times k^2 + C_{\text{in}} \times C_{\text{out}})$. The ratio is the same as the parameter ratio: $1/C_{\text{out}} + 1/k^2$. For 3x3 kernels with 64+ channels, this is roughly 8-9x reduction.</span>

- <span style="font-size: 14px;">**Q: What is the inverted residual in MobileNet v2 and why "inverted"?**</span>
  <span style="font-size: 14px;">A: A standard ResNet bottleneck reduces channels (256 to 64), convolves (64), then expands (64 to 256). The inverted residual does the opposite: expand (24 to 144 with expansion factor 6), depthwise conv (144), compress (144 to 24). It is "inverted" because the depthwise conv operates in a high-dimensional expanded space, giving it more capacity. The skip connection is on the narrow representation, saving memory.</span>

- <span style="font-size: 14px;">**Q: Why does MobileNet v2 use a linear bottleneck (no ReLU after the last pointwise conv)?**</span>
  <span style="font-size: 14px;">A: ReLU destroys information in low-dimensional spaces by zeroing negative values. When the narrow bottleneck has few channels, applying ReLU loses too much information. The paper shows that removing the final ReLU (making it a "linear bottleneck") preserves more information through the skip connection, improving accuracy significantly.</span>

- <span style="font-size: 14px;">**Q: How does ShuffleNet improve on depthwise separable convolutions?**</span>
  <span style="font-size: 14px;">A: ShuffleNet observes that the pointwise (1x1) convolution is the bottleneck in MobileNet (accounting for ~95% of computation). It replaces pointwise convs with grouped 1x1 convs, then adds a channel shuffle operation to enable cross-group information flow. This further reduces computation while maintaining accuracy.</span>

---
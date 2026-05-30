# <span style="font-size: 20px;">Encoder Block</span>

<span style="font-size: 14px;">The encoder block is the fundamental repeating unit of the contracting path in U-Net (Ronneberger, Fischer, and Brox, 2015), the fully convolutional network that set new benchmarks for biomedical image segmentation. Each encoder block applies two 3x3 unpadded convolutions followed by a 2x2 max pooling operation, progressively reducing spatial resolution while increasing feature channels. Understanding the exact spatial arithmetic is essential because the unpadded convolutions create dimension mismatches that propagate through the network and directly affect skip connections.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">The encoder block is the building block of U-Net's **contracting path** -- the left half of the U-shaped architecture. Each encoder block performs three operations in sequence:</span>

* <span style="font-size: 14px;">**First 3x3 convolution (unpadded) + ReLU:** Extracts local features from the input and increases the channel count to $C_{out}$. Because no padding is used, the spatial dimensions shrink by 2 (one pixel lost on each side).</span>
* <span style="font-size: 14px;">**Second 3x3 convolution (unpadded) + ReLU:** Further refines the feature representation at the same channel depth $C_{out}$. Again, spatial dimensions shrink by 2.</span>
* <span style="font-size: 14px;">**2x2 max pooling with stride 2:** Halves the spatial dimensions, serving as the downsampling step that increases the receptive field for the next block.</span>

<span style="font-size: 14px;">Critically, the encoder block produces **two outputs**: the pooled tensor (input to the next block) and the pre-pool feature map (saved as a **skip connection** for the decoder). The paper states: "The contracting path consists of the repeated application of two 3x3 convolutions (unpadded convolutions), each followed by a rectified linear unit (ReLU) and a 2x2 max pooling operation with stride 2 for downsampling." There are four such encoder blocks in the standard U-Net.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">The spatial output size of a convolution is governed by the standard formula:</span>

$$
H_{out} = \left\lfloor \frac{H_{in} + 2p - k}{s} \right\rfloor + 1
$$

<span style="font-size: 14px;">where $H_{in}$ is the input height, $p$ is padding, $k$ is kernel size, and $s$ is stride. For the U-Net encoder, the convolutions use $k = 3$, $p = 0$ (unpadded / valid mode), and $s = 1$. Substituting:</span>

$$
H_{out} = \left\lfloor \frac{H_{in} + 0 - 3}{1} \right\rfloor + 1 = H_{in} - 2
$$

<span style="font-size: 14px;">Each 3x3 unpadded convolution reduces the spatial dimension by exactly 2. Two consecutive convolutions therefore reduce it by 4. The same formula applies to the width dimension $W$.</span>

<span style="font-size: 14px;">For max pooling with kernel size 2 and stride 2:</span>

$$
H_{out} = \left\lfloor \frac{H_{in}}{2} \right\rfloor
$$

<span style="font-size: 14px;">Putting the full encoder block together, the shape transformation is:</span>

$$
\begin{array}{rcl}
\text{Input} & : & (B,\; H,\; W,\; C_{in}) \\
\xrightarrow{\text{Conv }3{\times}3} & : & (B,\; H{-}2,\; W{-}2,\; C_{out}) \\
\xrightarrow{\text{Conv }3{\times}3} & : & (B,\; H{-}4,\; W{-}4,\; C_{out}) \quad \leftarrow \text{skip connection} \\
\xrightarrow{\text{MaxPool }2{\times}2} & : & \left(B,\; \frac{H{-}4}{2},\; \frac{W{-}4}{2},\; C_{out}\right)
\end{array}
$$

<span style="font-size: 14px;">The channel dimension changes only at the first convolution (from $C_{in}$ to $C_{out}$). The second convolution maintains $C_{out}$, and max pooling preserves channels entirely -- it operates only on spatial dimensions.</span>

---

## <span style="font-size: 16px;">Why Unpadded Convolutions</span>

<span style="font-size: 14px;">The original U-Net uses **valid convolutions** ($p = 0$), meaning the kernel is only applied where it fully overlaps the input. No fabricated border values are introduced, so every output pixel is computed from real data. In biomedical imaging, where each output pixel is a segmentation decision, this avoids border artifacts that could corrupt cell boundary classifications.</span>

<span style="font-size: 14px;">The tradeoff is spatial shrinkage at every convolution layer. With two convolutions per block and four encoder blocks, the contracting path alone removes $4 \times 4 = 16$ pixels from each spatial side before accounting for pooling. This compounds through the bottleneck and decoder: a 572x572 input produces only a 388x388 output (92 pixels lost per side).</span>

<span style="font-size: 14px;">This shrinkage directly affects skip connections. Encoder feature maps are spatially larger than the corresponding decoder maps, so the decoder must **center-crop** encoder features before concatenation. Modern U-Net implementations switch to same-padding ($p = 1$) to avoid this, but the original unpadded design is why 572x572 was chosen as input size -- it guarantees even dimensions at every stage.</span>

---

## <span style="font-size: 16px;">The Two Outputs</span>

<span style="font-size: 14px;">Every encoder block produces two tensors, and both are essential for U-Net to function.</span>

<span style="font-size: 14px;">**The pooled tensor** is the output after max pooling. It has halved spatial dimensions and serves as the input to the next encoder block (or to the bottleneck, for the last encoder block). This tensor carries information deeper into the network, where each successive block operates at lower resolution but captures broader context.</span>

<span style="font-size: 14px;">**The pre-pool feature map** is the output after the second convolution but before max pooling. This tensor is saved and later used as a **skip connection** in the expanding path. It retains the full spatial resolution at that encoder level, preserving fine-grained localization information that pooling would discard.</span>

<span style="font-size: 14px;">Without skip connections, the decoder would reconstruct spatial detail entirely from the low-resolution bottleneck -- fine details are irrecoverably lost. Skip connections give the decoder direct access to the encoder's high-resolution features. During decoding, each decoder block upsamples its input, then crops and concatenates the matching encoder skip connection along the channel axis. The paper describes this as combining "high resolution features from the contracting path" with "the upsampled output."</span>

<span style="font-size: 14px;">A practical consequence: an implementation that returns only the pooled tensor (forgetting the pre-pool feature map) will silently break the decoder. Both outputs are required at every encoder level.</span>

---

## <span style="font-size: 16px;">The Channel Progression</span>

<span style="font-size: 14px;">The original U-Net follows a strict channel doubling pattern through the encoder. Starting from the input image (typically 1 channel for grayscale biomedical images), the four encoder blocks produce progressively more feature channels:</span>

* <span style="font-size: 14px;">**Encoder Block 1:** $1 \to 64$ channels -- maps raw pixel intensity to 64 feature maps capturing edges, gradients, and local textures.</span>
* <span style="font-size: 14px;">**Encoder Block 2:** $64 \to 128$ channels -- combines low-level features into corners, junctions, and texture patches.</span>
* <span style="font-size: 14px;">**Encoder Block 3:** $128 \to 256$ channels -- represents larger-scale structures like cell parts and tissue boundaries.</span>
* <span style="font-size: 14px;">**Encoder Block 4:** $256 \to 512$ channels -- captures the most abstract spatial features before the bottleneck.</span>

<span style="font-size: 14px;">After the four encoder blocks, the bottleneck operates at $512 \to 1024$ channels. The decoder then reverses the pattern: $1024 \to 512 \to 256 \to 128 \to 64$.</span>

<span style="font-size: 14px;">The doubling strategy is a deliberate tradeoff. Reducing spatial size by $2\times$ in each dimension reduces feature map area by $4\times$, so doubling the channel count only increases per-block parameters by roughly $2\times$ (since convolution weight count scales with $C_{in} \times C_{out} \times k^2$). The first block's jump ($1 \to 64$) is where the network transitions from raw pixels to learned representations; the subsequent doublings follow the standard VGGNet pattern.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">U-Net was introduced by Ronneberger, Fischer, and Brox in "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015). The paper addressed biomedical imaging, where training data is extremely scarce yet segmentation demands pixel-level precision. Building on the fully convolutional network (FCN) of Long et al. (2015), U-Net's key contribution was the symmetric encoder-decoder structure with skip connections at every level, dramatically improving localization accuracy.</span>

<span style="font-size: 14px;">The contracting path follows what Ronneberger et al. describe as "the typical architecture of a convolutional network." The design mirrors VGGNet's pattern of stacking 3x3 convolutions with 2x2 max pooling. The choice of 3x3 kernels throughout (rather than 5x5 or 7x7) was influenced by the VGGNet finding that stacking small kernels achieves the same receptive field with fewer parameters and more nonlinearities.</span>

<span style="font-size: 14px;">U-Net won the ISBI cell tracking challenge 2015 by a large margin (IOU 0.9203 on PhC-U373, 0.7756 on DIC-HeLa) with very limited training data. The paper also introduced the **overlap-tile strategy**: because unpadded convolutions make the output smaller than the input, adjacent tiles must overlap so the output covers the full image without gaps.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">The standard U-Net input is 572x572x1 (grayscale, single channel). Tracing through the first encoder block with $C_{out} = 64$:</span>

<span style="font-size: 14px;">**Input tensor shape:** $(1, 572, 572, 1)$</span>

<span style="font-size: 14px;">**After first 3x3 unpadded convolution** ($C_{in}=1, C_{out}=64$):</span>

$$
H = 572 - 2 = 570, \quad W = 572 - 2 = 570
$$

<span style="font-size: 14px;">Shape: $(1, 570, 570, 64)$. The channel count jumps from 1 to 64.</span>

<span style="font-size: 14px;">**After second 3x3 unpadded convolution** ($C_{in}=64, C_{out}=64$):</span>

$$
H = 570 - 2 = 568, \quad W = 570 - 2 = 568
$$

<span style="font-size: 14px;">Shape: $(1, 568, 568, 64)$. This is the **pre-pool feature map** -- the skip connection output saved for decoder block 4.</span>

<span style="font-size: 14px;">**After 2x2 max pooling** (stride 2):</span>

$$
H = \frac{568}{2} = 284, \quad W = \frac{568}{2} = 284
$$

<span style="font-size: 14px;">Shape: $(1, 284, 284, 64)$. This is the **pooled output** -- input to encoder block 2. Max pooling does not change the channel count.</span>

<span style="font-size: 14px;">So encoder block 1 produces:</span>

* <span style="font-size: 14px;">**Skip connection:** $(1, 568, 568, 64)$ -- saved for the decoder</span>
* <span style="font-size: 14px;">**Pooled output:** $(1, 284, 284, 64)$ -- input to encoder block 2</span>

---

## <span style="font-size: 16px;">The Contracting Path in Full</span>

<span style="font-size: 14px;">Tracing the full contracting path from 572x572 through all four encoder blocks. Each block: two 3x3 unpadded convolutions (-2 each) then 2x2 max pool (/2).</span>

<span style="font-size: 14px;">**Encoder Block 1** ($C_{in}=1, C_{out}=64$):</span>

$$
(1,\; 572,\; 572,\; 1) \xrightarrow{\text{conv}} (1,\; 570,\; 570,\; 64) \xrightarrow{\text{conv}} (1,\; 568,\; 568,\; 64) \xrightarrow{\text{pool}} (1,\; 284,\; 284,\; 64)
$$

* <span style="font-size: 14px;">Skip: $(1, 568, 568, 64)$</span>

<span style="font-size: 14px;">**Encoder Block 2** ($C_{in}=64, C_{out}=128$):</span>

$$
(1,\; 284,\; 284,\; 64) \xrightarrow{\text{conv}} (1,\; 282,\; 282,\; 128) \xrightarrow{\text{conv}} (1,\; 280,\; 280,\; 128) \xrightarrow{\text{pool}} (1,\; 140,\; 140,\; 128)
$$

* <span style="font-size: 14px;">Skip: $(1, 280, 280, 128)$</span>

<span style="font-size: 14px;">**Encoder Block 3** ($C_{in}=128, C_{out}=256$):</span>

$$
(1,\; 140,\; 140,\; 128) \xrightarrow{\text{conv}} (1,\; 138,\; 138,\; 256) \xrightarrow{\text{conv}} (1,\; 136,\; 136,\; 256) \xrightarrow{\text{pool}} (1,\; 68,\; 68,\; 256)
$$

* <span style="font-size: 14px;">Skip: $(1, 136, 136, 256)$</span>

<span style="font-size: 14px;">**Encoder Block 4** ($C_{in}=256, C_{out}=512$):</span>

$$
(1,\; 68,\; 68,\; 256) \xrightarrow{\text{conv}} (1,\; 66,\; 66,\; 512) \xrightarrow{\text{conv}} (1,\; 64,\; 64,\; 512) \xrightarrow{\text{pool}} (1,\; 32,\; 32,\; 512)
$$

* <span style="font-size: 14px;">Skip: $(1, 64, 64, 512)$</span>

<span style="font-size: 14px;">The pooled output of encoder block 4, $(1, 32, 32, 512)$, becomes the input to the bottleneck. The four skip connections at sizes $(568, 568)$, $(280, 280)$, $(136, 136)$, and $(64, 64)$ are later center-cropped and concatenated with the corresponding decoder blocks.</span>

<span style="font-size: 14px;">Notice that 572 was specifically chosen so all spatial dimensions remain even at every stage: $568 \to 284$, $280 \to 140$, $136 \to 68$, $64 \to 32$. If the input were 570, the first pool would yield $(570-4)/2 = 283$ (odd), causing rounding problems at subsequent pooling steps.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Forgetting the skip connection output.** The encoder block must return two tensors: the pooled tensor for the next block and the pre-pool feature map for the skip connection. An implementation that only returns the pooled output will silently break the decoder, which needs the skip connection for the crop-and-concatenate step. The decoder cannot recover fine-grained spatial information from the pooled tensor alone.</span>

* <span style="font-size: 14px;">**Wrong spatial reduction per convolution.** A 3x3 unpadded convolution reduces each spatial dimension by exactly 2, not by 3. The kernel size is 3, but the formula is $H_{out} = H_{in} - k + 1 = H_{in} - 2$ (not $H_{in} - 3$). This is because a 3x3 kernel centered at position 1 can still cover positions 0, 1, 2 -- it only loses 1 pixel on each side. Getting this wrong by even 1 pixel compounds across layers and causes shape mismatches deeper in the network.</span>

* <span style="font-size: 14px;">**Assuming pooling changes channels.** Max pooling operates independently on each channel, taking the maximum value in each $2 \times 2$ spatial window. It does not change the number of channels. The channel count changes only at convolution layers (specifically, at the first convolution of each block, where $C_{in} \to C_{out}$). An implementation that modifies channels during pooling will produce tensors with the wrong depth.</span>

* <span style="font-size: 14px;">**Odd spatial dimensions before pooling.** If the spatial dimension before a 2x2 max pool is odd (e.g., 283), floor division applies: $\lfloor 283/2 \rfloor = 141$, discarding one row or column. This asymmetric loss compounds through subsequent blocks. The original U-Net avoids this by choosing input size 572, which guarantees even dimensions at every pooling stage.</span>

* <span style="font-size: 14px;">**Confusing valid padding with same padding.** The original U-Net uses valid padding ($p = 0$), where each 3x3 convolution shrinks spatial dimensions by 2. Modern reimplementations often use same padding ($p = 1$). Mixing the two conventions -- valid padding in the encoder but same-padding assumptions in the decoder -- produces shape mismatches that crash at concatenation.</span>

* <span style="font-size: 14px;">**Applying the channel increase at the wrong convolution.** Both convolutions use the same $C_{out}$. The first maps $C_{in} \to C_{out}$, the second maps $C_{out} \to C_{out}$. Using different output channels for the two convolutions, or applying the channel increase only at the second, produces incorrect channel dimensions.</span>

* <span style="font-size: 14px;">**Forgetting that skip and decoder sizes differ.** Because of unpadded convolutions in the bottleneck and decoder, the encoder's skip connection is always spatially larger than the decoder feature map it will be concatenated with. For example, encoder block 4 saves a skip of $(64, 64)$, but the matching decoder map might be $(56, 56)$. The skip must be center-cropped before concatenation. Forgetting this crop produces misaligned features.</span>

---
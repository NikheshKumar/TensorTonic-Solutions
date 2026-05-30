# <span style="font-size: 20px;">Decoder Block</span>

<span style="font-size: 14px;">The decoder block is the fundamental building block of U-Net's expanding path. It reverses the encoder's spatial compression by upsampling feature maps, fusing them with high-resolution skip connections from the contracting path, and refining the result through two unpadded convolutions. Each decoder block recovers spatial detail that the encoder discarded, enabling the network to produce precise, pixel-level segmentation masks.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">The expanding path of U-Net is the mirror image of the contracting path. Where the encoder compresses spatial dimensions and increases channel depth, the decoder does the opposite: it expands spatial dimensions and decreases channel depth. The decoder block is the repeating unit that accomplishes this expansion.</span>

<span style="font-size: 14px;">Each decoder block performs exactly three operations in sequence:</span>

* <span style="font-size: 14px;">**Up-convolution (2x2, stride 2):** A learned transposed convolution that doubles the spatial dimensions and halves the number of channels.</span>
* <span style="font-size: 14px;">**Crop and concatenate:** The corresponding encoder feature map is center-cropped to match the upsampled spatial dimensions, then concatenated along the channel axis.</span>
* <span style="font-size: 14px;">**Two 3x3 unpadded convolutions:** Each convolution uses "valid" mode (no padding), shrinking the spatial dimensions by 2 per convolution, 4 total. Each is followed by ReLU.</span>

<span style="font-size: 14px;">The decoder block is how U-Net combines deep, semantically rich features from the bottleneck with fine-grained spatial detail from the encoder. Without it, the network would have semantic understanding but no spatial precision.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">Let the decoder block input have shape $(B, H, W, C)$ and the corresponding encoder skip connection have shape $(B, H_e, W_e, C/2)$.</span>

<span style="font-size: 14px;">**Step 1 -- Up-convolution (transposed conv, 2x2, stride 2):**</span>

$$
(B, H, W, C) \xrightarrow{\text{Up-conv}} (B, 2H, 2W, C/2)
$$

<span style="font-size: 14px;">The spatial dimensions double and the channel count halves.</span>

<span style="font-size: 14px;">**Step 2 -- Crop and concatenate:**</span>

$$
\text{crop}(H_e, W_e \to 2H, 2W) \implies (B, 2H, 2W, C/2)
$$

$$
\text{concat}\bigl((B, 2H, 2W, C/2),\; (B, 2H, 2W, C/2)\bigr) = (B, 2H, 2W, C)
$$

<span style="font-size: 14px;">The encoder feature map is center-cropped to match the upsampled decoder map, then concatenated along the channel axis, doubling channels from $C/2$ to $C$.</span>

<span style="font-size: 14px;">**Step 3 -- Two 3x3 unpadded convolutions:**</span>

$$
(B, 2H, 2W, C) \xrightarrow{\text{Conv }3\times3} (B, 2H-2, 2W-2, C_{out}) \xrightarrow{\text{Conv }3\times3} (B, 2H-4, 2W-4, C_{out})
$$

<span style="font-size: 14px;">Each valid convolution reduces spatial dimensions by 2. After two, the total reduction is 4. The output channel count $C_{out}$ is typically $C/2$.</span>

<span style="font-size: 14px;">**General spatial formula for valid convolution:**</span>

$$
H_{out} = H_{in} - k + 1
$$

<span style="font-size: 14px;">For $k = 3$: $H_{out} = H_{in} - 2$. Applied twice: $H_{out} = H_{in} - 4$.</span>

---

## <span style="font-size: 16px;">The Three Operations</span>

### <span style="font-size: 14px;">Operation 1: Up-Convolution</span>

<span style="font-size: 14px;">The up-convolution (transposed convolution) is the inverse of a strided convolution. A standard 2x2 convolution with stride 2 maps $2H \times 2W$ to $H \times W$. The up-convolution reverses this: $H \times W$ becomes $2H \times 2W$.</span>

<span style="font-size: 14px;">Each input pixel is multiplied by the $2 \times 2$ kernel to produce a $2 \times 2$ patch in the output. Adjacent input pixels produce adjacent patches with no overlap (stride equals kernel size), so the output is exactly twice the spatial size of the input.</span>

<span style="font-size: 14px;">The transposed convolution formula confirms this:</span>

$$
H_{out} = (H_{in} - 1) \times s + k = (H_{in} - 1) \times 2 + 2 = 2H_{in}
$$

<span style="font-size: 14px;">The channel count halves because the up-convolution is configured with $C/2$ output filters. This is deliberate: the skip connection will add $C/2$ more channels via concatenation, so the combined result has $C$ channels for the subsequent convolutions.</span>

<span style="font-size: 14px;">Why not bilinear interpolation? Bilinear upsampling is parameter-free but cannot learn task-specific upsampling patterns. The transposed convolution has $2 \times 2 \times C \times C/2$ learnable parameters that the network optimizes for the segmentation task.</span>

### <span style="font-size: 14px;">Operation 2: Crop and Concatenate</span>

<span style="font-size: 14px;">After upsampling, the decoder feature map must be merged with the corresponding encoder feature map. The encoder map carries spatial detail (edges, textures, fine boundaries) that was lost during max pooling. The decoder map carries semantic context built up through the bottleneck.</span>

<span style="font-size: 14px;">The encoder feature map is always spatially larger than the upsampled decoder map. This mismatch arises because unpadded convolutions in both paths shrink spatial dimensions at every stage. The encoder's pre-pooling feature map was saved after its two convolutions but before pooling, so it retains more spatial extent than the decoder path at the corresponding level.</span>

<span style="font-size: 14px;">The encoder map is center-cropped: equal pixels are removed from each border so the center region matches the upsampled decoder's spatial size. The crop per side is:</span>

$$
\text{crop}_{\text{per\_side}} = \frac{H_e - 2H}{2}
$$

<span style="font-size: 14px;">After cropping, the two maps have identical spatial dimensions and are concatenated along the channel axis, doubling the channel count. This concatenation (not addition) preserves both feature sets independently and lets the subsequent convolutions learn how to combine them.</span>

### <span style="font-size: 14px;">Operation 3: Two 3x3 Unpadded Convolutions</span>

<span style="font-size: 14px;">The concatenated feature map passes through two consecutive 3x3 convolutions, each followed by ReLU. These convolutions serve two purposes:</span>

* <span style="font-size: 14px;">**Feature integration:** Learn to combine the encoder's spatial detail with the decoder's semantic context from the concatenated channels.</span>
* <span style="font-size: 14px;">**Channel reduction:** Map from $C$ input channels to $C_{out}$ output channels (typically $C/2$), restoring the channel count for the next decoder block.</span>

<span style="font-size: 14px;">Because these convolutions use no padding, each one shrinks spatial dimensions by 2. Two convolutions shrink by 4 total, matching the behavior of encoder blocks.</span>

---

## <span style="font-size: 16px;">The Skip Connection Integration</span>

<span style="font-size: 14px;">The skip connection is U-Net's defining architectural innovation. Without it, the decoder would reconstruct spatial detail from the bottleneck alone, producing blurry segmentation boundaries. With skip connections, the decoder has direct access to the encoder's high-resolution feature maps at every scale.</span>

<span style="font-size: 14px;">At each encoder level, the feature map after the two 3x3 convolutions (but before max pooling) is saved. This pre-pooling map preserves fine spatial detail that max pooling would destroy. When the decoder reaches the corresponding level, this saved map is retrieved and merged via crop+concat.</span>

<span style="font-size: 14px;">The paper states: "a concatenation with the correspondingly cropped feature map from the contracting path." The word "correspondingly" means each decoder level is matched with its mirror encoder level. Level 1 of the decoder receives the skip from encoder level 4, level 2 from level 3, and so on -- symmetric across the bottleneck.</span>

<span style="font-size: 14px;">U-Net uses concatenation rather than element-wise addition (as in ResNet). Concatenation preserves both feature sets as separate channel groups. Addition forces features into the same representational space immediately. Concatenation gives the subsequent convolutions more raw material, at the cost of temporarily doubling the channel count.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">U-Net was introduced by Ronneberger, Fischer, and Brox (2015) for biomedical image segmentation. The core insight was that medical segmentation requires both semantic understanding (what is this tissue?) and precise spatial localization (where exactly is the boundary?). The encoder-decoder structure with skip connections delivers both.</span>

<span style="font-size: 14px;">The expanding path mirrors the contracting path: four encoder blocks with two 3x3 convolutions + max pooling are matched by four decoder blocks with up-conv + crop+concat + two 3x3 convolutions. The channel progression reverses:</span>

* <span style="font-size: 14px;">**Contracting:** 1 -> 64 -> 128 -> 256 -> 512 -> 1024 (bottleneck)</span>
* <span style="font-size: 14px;">**Expanding:** 1024 -> 512 -> 256 -> 128 -> 64</span>

<span style="font-size: 14px;">Each decoder block's up-conv halves the channels, the skip connection doubles them via concat, and the two convolutions restore them to the target count. The symmetry is precise: channels at each decoder level match the corresponding encoder level.</span>

<span style="font-size: 14px;">U-Net achieved state-of-the-art results on the ISBI cell tracking challenge with only 30 annotated training images. The skip connections were critical to this data efficiency -- by reusing encoder features directly, the decoder avoided relearning spatial patterns from scratch.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Trace decoder block 1 (immediately after the bottleneck). Batch size $B = 1$.</span>

<span style="font-size: 14px;">**Input to the decoder block (from bottleneck):** $(1, 28, 28, 1024)$</span>

<span style="font-size: 14px;">**Encoder skip connection (from encoder level 4):** $(1, 64, 64, 512)$</span>

<span style="font-size: 14px;">**Step 1 -- Up-convolution (2x2, stride 2, 512 filters):**</span>

<span style="font-size: 14px;">Spatial doubles: $28 \times 2 = 56$. Channels halve: $1024 / 2 = 512$.</span>

$$
(1, 28, 28, 1024) \to (1, 56, 56, 512)
$$

<span style="font-size: 14px;">**Step 2 -- Crop encoder features:**</span>

<span style="font-size: 14px;">Encoder spatial: $64$. Decoder spatial: $56$. Crop per side: $(64 - 56) / 2 = 4$.</span>

$$
(1, 64, 64, 512) \xrightarrow{\text{crop}} (1, 56, 56, 512)
$$

<span style="font-size: 14px;">**Step 3 -- Concatenate along channel axis:**</span>

$$
\text{concat}\bigl((1, 56, 56, 512),\; (1, 56, 56, 512)\bigr) = (1, 56, 56, 1024)
$$

<span style="font-size: 14px;">**Step 4 -- First 3x3 unpadded convolution (512 filters):**</span>

$$
(1, 56, 56, 1024) \to (1, 54, 54, 512)
$$

<span style="font-size: 14px;">**Step 5 -- Second 3x3 unpadded convolution (512 filters):**</span>

$$
(1, 54, 54, 512) \to (1, 52, 52, 512)
$$

<span style="font-size: 14px;">**Final output of decoder block 1:** $(1, 52, 52, 512)$.</span>

---

## <span style="font-size: 16px;">The Expanding Path in Full</span>

<span style="font-size: 14px;">All four decoder blocks traced end-to-end. Spatial values follow the original paper's $572 \times 572$ input.</span>

<span style="font-size: 14px;">**Bottleneck output:** $(1, 28, 28, 1024)$</span>

<span style="font-size: 14px;">**Decoder Block 1** (encoder skip: $(1, 64, 64, 512)$):</span>

* <span style="font-size: 14px;">Up-conv: $(1, 28, 28, 1024) \to (1, 56, 56, 512)$</span>
* <span style="font-size: 14px;">Crop encoder: $(1, 64, 64, 512) \to (1, 56, 56, 512)$</span>
* <span style="font-size: 14px;">Concat: $(1, 56, 56, 1024)$</span>
* <span style="font-size: 14px;">Conv 1: $(1, 56, 56, 1024) \to (1, 54, 54, 512)$</span>
* <span style="font-size: 14px;">Conv 2: $(1, 54, 54, 512) \to (1, 52, 52, 512)$</span>

<span style="font-size: 14px;">**Decoder Block 2** (encoder skip: $(1, 136, 136, 256)$):</span>

* <span style="font-size: 14px;">Up-conv: $(1, 52, 52, 512) \to (1, 104, 104, 256)$</span>
* <span style="font-size: 14px;">Crop encoder: $(1, 136, 136, 256) \to (1, 104, 104, 256)$</span>
* <span style="font-size: 14px;">Concat: $(1, 104, 104, 512)$</span>
* <span style="font-size: 14px;">Conv 1: $(1, 104, 104, 512) \to (1, 102, 102, 256)$</span>
* <span style="font-size: 14px;">Conv 2: $(1, 102, 102, 256) \to (1, 100, 100, 256)$</span>

<span style="font-size: 14px;">**Decoder Block 3** (encoder skip: $(1, 280, 280, 128)$):</span>

* <span style="font-size: 14px;">Up-conv: $(1, 100, 100, 256) \to (1, 200, 200, 128)$</span>
* <span style="font-size: 14px;">Crop encoder: $(1, 280, 280, 128) \to (1, 200, 200, 128)$</span>
* <span style="font-size: 14px;">Concat: $(1, 200, 200, 256)$</span>
* <span style="font-size: 14px;">Conv 1: $(1, 200, 200, 256) \to (1, 198, 198, 128)$</span>
* <span style="font-size: 14px;">Conv 2: $(1, 198, 198, 128) \to (1, 196, 196, 128)$</span>

<span style="font-size: 14px;">**Decoder Block 4** (encoder skip: $(1, 568, 568, 64)$):</span>

* <span style="font-size: 14px;">Up-conv: $(1, 196, 196, 128) \to (1, 392, 392, 64)$</span>
* <span style="font-size: 14px;">Crop encoder: $(1, 568, 568, 64) \to (1, 392, 392, 64)$</span>
* <span style="font-size: 14px;">Concat: $(1, 392, 392, 128)$</span>
* <span style="font-size: 14px;">Conv 1: $(1, 392, 392, 128) \to (1, 390, 390, 64)$</span>
* <span style="font-size: 14px;">Conv 2: $(1, 390, 390, 64) \to (1, 388, 388, 64)$</span>

<span style="font-size: 14px;">The final output $(1, 388, 388, 64)$ feeds into a $1 \times 1$ convolution that maps 64 channels to the number of segmentation classes. The output is $388 \times 388$, smaller than the $572 \times 572$ input due to unpadded convolutions throughout the network.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

<span style="font-size: 14px;">**1. Wrong up-convolution output dimensions.** With kernel size 2 and stride 2, the output is exactly $2H \times 2W$. Using the general formula $(H-1) \times s + k$ with wrong $k$ or $s$ (e.g., $k=3$ instead of $k=2$) produces incorrect spatial dimensions that cascade through the rest of the block.</span>

<span style="font-size: 14px;">**2. Forgetting to crop the encoder features.** The encoder skip connection is always spatially larger than the upsampled decoder map. Attempting to concatenate without cropping fails due to mismatched spatial dimensions. The crop applies to the encoder map (not the decoder map) and must be a center crop (equal pixels removed from each border).</span>

<span style="font-size: 14px;">**3. Wrong concatenation axis.** Concatenation must happen along the channel axis, not spatial axes. If both maps are $(1, 56, 56, 512)$, the result is $(1, 56, 56, 1024)$. Concatenating along height would give $(1, 112, 56, 512)$, which is wrong.</span>

<span style="font-size: 14px;">**4. Spatial mismatch after crop.** The crop amount must be computed precisely: $(H_e - 2H) / 2$ pixels per side. An off-by-one error (cropping 3 on one side and 5 on the other) causes the concatenation to fail or produces misaligned features.</span>

<span style="font-size: 14px;">**5. Forgetting that two convolutions shrink spatial dimensions.** Each 3x3 unpadded convolution reduces spatial dimensions by 2 (not 1). Two convolutions reduce by 4 total. A common mistake is assuming padded convolutions (which preserve spatial size) or forgetting the second convolution entirely.</span>

<span style="font-size: 14px;">**6. Confusing channel counts through the block.** The channel count changes three times: halved by up-conv ($C \to C/2$), doubled by concat ($C/2 \to C$), then reduced by convolutions ($C \to C_{out}$, typically $C/2$). Losing track of which operation changes channels is a frequent source of shape errors.</span>

<span style="font-size: 14px;">**7. Applying operations in wrong order.** The sequence must be: up-conv, then crop+concat, then two convolutions. Applying convolutions before the skip connection or concatenating before upsampling produces entirely different architectures.</span>

---
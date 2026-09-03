# <span style="font-size: 20px;">Patch Embedding</span>

<span style="font-size: 14px;">Patch embedding is the input transformation that converts a raw image into a sequence of token vectors suitable for a Transformer encoder. Introduced in the Vision Transformer (Dosovitskiy et al., 2020), it splits an image into fixed-size non-overlapping patches, flattens each patch into a vector, and linearly projects that vector into the model's embedding dimension, producing the same kind of token sequence that NLP Transformers expect from text.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">A standard Transformer encoder processes a sequence of $N$ vectors, each of dimension $D$. In NLP, these vectors come from a token embedding lookup table. In vision, there are no discrete tokens. The image is a dense grid of pixel values with shape $(B, H, W, C)$ where $B$ is batch size, $H$ is height, $W$ is width, and $C$ is the number of channels (3 for RGB). Patch embedding bridges this gap by converting the spatial image into a flat sequence of **patch tokens**.</span>

<span style="font-size: 14px;">The operation proceeds in three stages. First, the image is divided into a grid of non-overlapping square patches of size $P \times P$. Second, each patch is flattened from shape $(P, P, C)$ into a vector of length $P^2 \cdot C$. Third, each flattened vector is linearly projected to dimension $D$ using a learned weight matrix. The result is $N = (H/P) \times (W/P)$ vectors in $\mathbb{R}^D$, ready to be fed into the Transformer encoder alongside a class token and position embeddings.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

### <span style="font-size: 14px;">Patch Count</span>

<span style="font-size: 14px;">The total number of patches extracted from an image of spatial size $H \times W$ using patch size $P$ is:</span>

$$
N = \frac{H}{P} \times \frac{W}{P}
$$

<span style="font-size: 14px;">This requires that $H$ and $W$ are both exactly divisible by $P$. If they are not, the image must be resized or padded before patch extraction.</span>

### <span style="font-size: 14px;">Reshape and Flatten</span>

<span style="font-size: 14px;">The input image tensor of shape $(B, H, W, C)$ is reshaped into a sequence of flattened patches:</span>

$$
(B, H, W, C) \to \left(B, \frac{H}{P}, P, \frac{W}{P}, P, C\right) \to \left(B, \frac{H}{P} \times \frac{W}{P}, P^2 \cdot C\right)
$$

<span style="font-size: 14px;">The first reshape splits height into $H/P$ blocks of $P$ rows and width into $W/P$ blocks of $P$ columns. A transpose groups all spatial elements of each patch together, and the final reshape collapses them into a single vector of length $P^2 \cdot C$.</span>

### <span style="font-size: 14px;">Linear Projection</span>

<span style="font-size: 14px;">Each flattened patch vector $x_p \in \mathbb{R}^{P^2 \cdot C}$ is projected into the model's embedding space using a learned weight matrix $W_{\text{proj}} \in \mathbb{R}^{(P^2 \cdot C) \times D}$ and a bias vector $b_{\text{proj}} \in \mathbb{R}^D$:</span>

$$
z_p = x_p \, W_{\text{proj}} + b_{\text{proj}}
$$

<span style="font-size: 14px;">Applied to all $N$ patches, this produces a tensor of shape $(B, N, D)$ that serves as the initial token sequence for the Transformer encoder.</span>

---

## <span style="font-size: 16px;">The Patch Grid</span>

<span style="font-size: 14px;">The image is tiled into a regular grid of non-overlapping $P \times P$ patches with $H/P$ rows and $W/P$ columns. Every pixel belongs to exactly one patch, and no pixel is shared between adjacent patches or left uncovered.</span>

<span style="font-size: 14px;">The patches are read in raster order: left to right across each row, top to bottom across rows. Patch index 0 corresponds to the top-left corner, patch index $W/P - 1$ to the top-right, and patch index $N - 1$ to the bottom-right. This ordering defines the position indices used by the subsequent position embedding layer.</span>

<span style="font-size: 14px;">Each patch captures a local spatial region. For ViT-B/16 with $P = 16$ on a $224 \times 224$ input, each patch covers a $16 \times 16$ pixel region containing $16 \times 16 \times 3 = 768$ raw values. The grid has $14 \times 14 = 196$ patches, so the Transformer processes a 196-token sequence rather than the 50,176-token sequence that pixel-level tokenization would produce.</span>

<span style="font-size: 14px;">The non-overlapping constraint is critical. Overlapping patches would increase $N$ beyond $(H/P) \times (W/P)$ and cause the same pixel information to appear in multiple tokens, creating redundancy. Non-overlapping patches provide a clean, minimal partition of the image into disjoint regions.</span>

---

## <span style="font-size: 16px;">Flatten Then Project</span>

<span style="font-size: 14px;">Once each $P \times P$ patch is isolated from the grid, it must be converted into a single vector that the Transformer can process as a token. This happens in two steps.</span>

### <span style="font-size: 14px;">Flattening</span>

<span style="font-size: 14px;">Each patch is a 3D tensor of shape $(P, P, C)$. Flattening collapses all three dimensions into a single vector of length $P^2 \cdot C$. For RGB images with $C = 3$ and $P = 16$, this produces a vector of length $16^2 \times 3 = 768$. The flattening order must be consistent across all patches: typically row-major within each channel, then channels concatenated. Note that $P^2 \cdot C$ includes all channels; forgetting $C$ would yield vectors of length $P^2 = 256$ instead of the correct $768$.</span>

### <span style="font-size: 14px;">Linear Projection</span>

<span style="font-size: 14px;">The flattened vector is generally not the same size as the model dimension $D$, and even when it matches numerically, a learned projection is still necessary. Raw pixel values are not meaningful token representations. The projection $W_{\text{proj}} \in \mathbb{R}^{(P^2 \cdot C) \times D}$ learns to map raw patch pixels into a space where attention can operate effectively.</span>

<span style="font-size: 14px;">In ViT-B/16, the flattened dimension is $768$ and $D = 768$, so $W_{\text{proj}}$ happens to be square. This is a coincidence of configuration. In ViT-L/16, $D = 1024$ while $P^2 \cdot C = 768$, so $W_{\text{proj}}$ has shape $(768, 1024)$. The projection dimension is always determined by the model configuration, not by the patch size.</span>

---

## <span style="font-size: 16px;">Why Not Just Use Pixels</span>

<span style="font-size: 14px;">The fundamental motivation for patch embedding is computational. Self-attention has complexity $O(N^2 \cdot D)$ where $N$ is the sequence length. If each pixel were a separate token, a $224 \times 224$ image would produce $N = 50{,}176$ tokens. The attention matrix alone would contain $50{,}176^2 \approx 2.5$ billion entries, which is completely impractical.</span>

<span style="font-size: 14px;">With $P = 16$, the same image yields $N = 14 \times 14 = 196$ tokens. The attention matrix shrinks to $196 \times 196 = 38{,}416$ entries. The sequence length reduction factor is $P^2 = 256$, and since attention cost is $O(N^2)$, the compute saving is roughly $P^4 = 65{,}536\times$ compared to pixel-level tokenization.</span>

<span style="font-size: 14px;">The trade-off is that each patch token represents a $16 \times 16$ spatial region, so the Transformer cannot attend to individual pixels directly. Intra-patch structure is captured only through the linear projection. The Transformer reasons about inter-patch relationships via attention, while intra-patch processing is limited to this single linear layer. Dosovitskiy et al. found this trade-off effective for image classification, and the approach scales well with larger datasets and model sizes.</span>

---

## <span style="font-size: 16px;">The Convolution Equivalence</span>

<span style="font-size: 14px;">The entire patch embedding operation can be implemented as a single 2D convolution with kernel size equal to $P$ and stride equal to $P$. Specifically, a `Conv2d` layer with $D$ output channels, kernel size $(P, P)$, and stride $(P, P)$ applied to the input image produces exactly the same result as the reshape-flatten-project pipeline.</span>

<span style="font-size: 14px;">The convolution slides a $P \times P$ kernel across the image with stride $P$, so each application covers exactly one non-overlapping patch. With $D$ output filters, the convolution produces $D$ values at each of the $(H/P) \times (W/P)$ spatial positions. Reshaping the output from $(B, H/P, W/P, D)$ to $(B, N, D)$ gives the identical sequence of patch embeddings.</span>

<span style="font-size: 14px;">This equivalence is not an approximation. The convolutional formulation is mathematically identical to the reshape-flatten-project pipeline. The convolution approach is preferred in practice because deep learning frameworks have highly optimized Conv2d implementations that exploit memory layout and hardware parallelism. Dosovitskiy et al. describe the operation as a linear projection of flattened patches in the paper, but the official JAX implementation uses a convolutional layer. Most PyTorch implementations follow suit, using `nn.Conv2d(in_channels=C, out_channels=D, kernel_size=P, stride=P)` as the patch embedding layer.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">The Vision Transformer was introduced by Dosovitskiy et al. in "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (2020). The title itself references patch embedding: the "16x16 words" are $16 \times 16$ pixel patches that serve as visual tokens analogous to word tokens in NLP. The paper describes it concisely: "We split an image into fixed-size patches, linearly embed each of them." Patch embedding is the minimal adaptation needed to convert a 2D image into a 1D token sequence for a standard Transformer.</span>

### <span style="font-size: 14px;">ViT Configurations</span>

<span style="font-size: 14px;">The paper defines several model configurations. The most commonly referenced are:</span>

* <span style="font-size: 14px;">**ViT-B/16:** $P = 16$, $D = 768$, 12 Transformer layers, 12 attention heads. Input $224 \times 224$ produces $N = 196$ patches.</span>
* <span style="font-size: 14px;">**ViT-L/16:** $P = 16$, $D = 1024$, 24 Transformer layers, 16 attention heads. Input $224 \times 224$ produces $N = 196$ patches.</span>
* <span style="font-size: 14px;">**ViT-B/32:** $P = 32$, $D = 768$, 12 Transformer layers, 12 attention heads. Input $224 \times 224$ produces $N = 49$ patches.</span>

<span style="font-size: 14px;">The notation "ViT-B/16" encodes the model size (Base) and patch size (16). Larger patches produce shorter sequences, reducing computational cost but also reducing spatial granularity. ViT-B/32 with only 49 tokens is significantly cheaper than ViT-B/16 with 196 tokens, but the coarser patches lose fine spatial detail.</span>

### <span style="font-size: 14px;">Design Decision</span>

<span style="font-size: 14px;">The patch embedding is intentionally simple. The authors' goal was to demonstrate that a standard Transformer applied to image patches with minimal modifications could achieve competitive classification results when trained on sufficient data. The linear projection from flattened patches is the simplest possible tokenization for images. No convolutional feature extraction, no multi-scale processing, no hierarchical pooling. Just split, flatten, project. This simplicity is the paper's central design principle: "we apply a standard Transformer directly to images, with the fewest possible modifications."</span>

---

## Numerical Example ($224 \times 224 \times 3$, $P = 16$, $D = 768$)

<span style="font-size: 14px;">Walk through the patch embedding computation for the standard ViT-B/16 configuration.</span>

<span style="font-size: 14px;">**Input image shape:** $(B, 224, 224, 3)$. A batch of RGB images at $224 \times 224$ resolution.</span>

<span style="font-size: 14px;">1. **Compute the number of patches.** The grid has $224 / 16 = 14$ rows and $224 / 16 = 14$ columns. The total patch count is:</span>

$$
N = 14 \times 14 = 196
$$

<span style="font-size: 14px;">2. **Extract and flatten patches.** Each patch covers a $16 \times 16$ spatial region across all 3 channels. The flattened dimension per patch is:</span>

$$
P^2 \cdot C = 16^2 \times 3 = 256 \times 3 = 768
$$

<span style="font-size: 14px;">After extracting and flattening, the tensor shape is $(B, 196, 768)$. Each of the 196 rows is a 768-dimensional vector of raw pixel values from one patch.</span>

<span style="font-size: 14px;">3. **Linear projection.** The weight matrix $W_{\text{proj}}$ has shape $(768, 768)$ and the bias $b_{\text{proj}}$ has shape $(768,)$. Each patch vector is projected:</span>

$$
z_p = x_p \, W_{\text{proj}} + b_{\text{proj}} \in \mathbb{R}^{768}
$$

<span style="font-size: 14px;">The output tensor has shape $(B, 196, 768)$. In this particular configuration, the input and output dimensions of the projection happen to be equal (both 768), but this is not true in general.</span>

<span style="font-size: 14px;">4. **Parameter count.** The patch embedding layer contains $768 \times 768 + 768 = 590{,}592$ parameters from the weight matrix and bias. This is a small fraction of ViT-B's total 86 million parameters.</span>

<span style="font-size: 14px;">5. **A concrete patch.** Consider patch (0, 0), the top-left patch covering rows 0-15 and columns 0-15. Its raw values form a tensor of shape $(16, 16, 3)$. Flattening yields a 768-element vector: the first 256 elements are red channel values, the next 256 are green, and the final 256 are blue. This vector is multiplied by $W_{\text{proj}}$ and added to $b_{\text{proj}}$ to produce the 768-dimensional embedding for this patch.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Image dimensions not divisible by patch size.** If $H$ or $W$ is not exactly divisible by $P$, the image cannot be cleanly tiled into non-overlapping patches. Attempting to reshape will fail with a dimension mismatch. For example, a $220 \times 220$ image with $P = 16$ gives $220 / 16 = 13.75$, which is not an integer. The image must be resized to $224 \times 224$ (or another $P$-divisible resolution) before patch extraction. This is a hard constraint, not a soft one.</span>
* <span style="font-size: 14px;">**Wrong reshape order.** A naive reshape that treats the image as a flat vector and splits into chunks of size $P^2 \cdot C$ will mix pixels from different patches. The correct approach is to reshape to $(B, H/P, P, W/P, P, C)$, transpose to group the two $P$ dimensions together, then flatten to $(B, N, P^2 \cdot C)$. Incorrect ordering produces patches that span non-contiguous image regions.</span>
* <span style="font-size: 14px;">**Confusing $P^2 \cdot C$ with $P^2$.** The flattened patch dimension is $P^2 \cdot C$, not $P^2$. For RGB images, each patch has 3 channels, so the flattened length is $3\times$ larger than the spatial element count. Using $P^2$ as the input dimension for the linear projection produces a dimension mismatch. For $P = 16$ and $C = 3$, the correct input dimension is $768$, not $256$.</span>
* <span style="font-size: 14px;">**Wrong projection matrix dimensions.** The weight matrix $W_{\text{proj}}$ maps from $P^2 \cdot C$ to $D$, so its shape is $(P^2 \cdot C, D)$. Transposing this to $(D, P^2 \cdot C)$ either crashes on a dimension mismatch or silently produces embeddings in the wrong space. When using the Conv2d equivalence the framework handles weight layout internally, but manual implementations must get this right.</span>
* <span style="font-size: 14px;">**Forgetting the batch dimension in reshape.** The reshape must preserve the batch dimension as the leading axis. Reshaping $(B, H, W, C)$ without accounting for $B$ leads to patches from different images being mixed together, producing silently wrong embeddings without raising any error.</span>
* <span style="font-size: 14px;">**Assuming $D = P^2 \cdot C$ always.** In ViT-B/16, the flattened dimension ($768$) coincidentally equals $D$ ($768$). This is a special case. In ViT-L/16, $D = 1024$ while $P^2 \cdot C = 768$, so $W_{\text{proj}}$ is rectangular. Hardcoding the assumption that input and output dimensions match will break for non-Base configurations.</span>
* <span style="font-size: 14px;">**Confusing patch embedding with position embedding.** Patch embedding converts image patches into vectors. Position embedding adds learned spatial information so the Transformer knows where each patch originated. These are separate sequential operations. Omitting either one yields a model that lacks spatial awareness or receives raw pixel vectors instead of learned representations.</span>

---
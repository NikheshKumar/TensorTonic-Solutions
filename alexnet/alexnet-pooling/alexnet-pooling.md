# <span style="font-size: 20px;">Overlapping Max Pooling</span>

<span style="font-size: 14px;">Pooling is a downsampling operation in CNNs that reduces spatial dimensions of feature maps by summarizing local regions into single values. In AlexNet (Krizhevsky, Sutskever, and Hinton, 2012), overlapping max pooling was a deliberate architectural choice that contributed to the network's landmark top-5 error rate of 15.3% on ILSVRC-2012.</span>

---

## <span style="font-size: 16px;">What Pooling Does</span>

<span style="font-size: 14px;">Pooling slides a fixed-size window across an input feature map and computes a single summary value per position, producing a smaller output that retains the most important information while discarding fine-grained spatial detail.</span>

<span style="font-size: 14px;">Key purposes:</span>

* <span style="font-size: 14px;">**Spatial downsampling:** Reduces height and width, decreasing activations for subsequent layers. In AlexNet, pooling reduces dimensions from 55x55 to 6x6.</span>
* <span style="font-size: 14px;">**Translation invariance:** Small spatial shifts produce similar pooled outputs, helping generalization.</span>
* <span style="font-size: 14px;">**Reducing computational cost:** Smaller feature maps mean fewer operations in subsequent layers.</span>
* <span style="font-size: 14px;">**Controlling overfitting:** Summarizing local regions acts as regularization. The AlexNet paper notes overlapping pooling made the model "slightly more difficult to overfit."</span>

<span style="font-size: 14px;">Pooling operates independently per channel, so channel count is unchanged. Only spatial dimensions (height and width) are reduced.</span>

---

## <span style="font-size: 16px;">Max Pooling vs Average Pooling</span>

### <span style="font-size: 14px;">Max Pooling</span>

<span style="font-size: 14px;">Selects the largest value from each $k \times k$ window:</span>

$$
y_{i,j} = \max_{0 \leq m < k, \; 0 \leq n < k} x_{i+m, \; j+n}
$$

<span style="font-size: 14px;">During backpropagation, the gradient flows only to the position that held the maximum; all others receive zero gradient.</span>

### <span style="font-size: 14px;">Average Pooling</span>

<span style="font-size: 14px;">Computes the arithmetic mean of all values within each window:</span>

$$
y_{i,j} = \frac{1}{k^2} \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} x_{i+m, \; j+n}
$$

<span style="font-size: 14px;">Distributes gradient equally to all positions during backpropagation.</span>

### <span style="font-size: 14px;">Why AlexNet Used Max Pooling</span>

* <span style="font-size: 14px;">**Preserves strongest activations:** After ReLU, the most informative signal is the magnitude of the strongest activation. Average pooling dilutes it by blending in weaker or zero activations.</span>
* <span style="font-size: 14px;">**Better gradient flow:** Gradients flow only through the max position, strengthening the most discriminative features.</span>
* <span style="font-size: 14px;">**Robustness to sparse activations:** After ReLU, many activations are zero. Max pooling ignores zeros; average pooling is dragged down by them.</span>
* <span style="font-size: 14px;">**Empirical performance:** Max pooling consistently outperformed average pooling on image classification benchmarks of that era.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

### <span style="font-size: 14px;">Output Dimension Formula</span>

<span style="font-size: 14px;">For input size $H_{in} \times W_{in}$, kernel size $k$, and stride $s$ with no padding:</span>

$$
H_{out} = \left\lfloor \frac{H_{in} - k}{s} \right\rfloor + 1
$$

$$
W_{out} = \left\lfloor \frac{W_{in} - k}{s} \right\rfloor + 1
$$

<span style="font-size: 14px;">AlexNet pooling always uses $p = 0$. The convolution formula includes a $+2p$ term in the numerator; do not confuse the two.</span>

### <span style="font-size: 14px;">Max Pooling Operation</span>

<span style="font-size: 14px;">For input $X$ with $C$ channels, the output at position $(i, j)$ for channel $c$:</span>

$$
Y(c, i, j) = \max_{0 \leq m < k, \; 0 \leq n < k} X(c, \; i \cdot s + m, \; j \cdot s + n)
$$

<span style="font-size: 14px;">The channel index $c$ is the same in input and output since pooling operates per channel independently.</span>

### <span style="font-size: 14px;">Full Output Shape</span>

<span style="font-size: 14px;">Input $(N, C, H_{in}, W_{in})$ produces output:</span>

$$
(N, \; C, \; H_{out}, \; W_{out})
$$

<span style="font-size: 14px;">Batch size $N$ and channel count $C$ are unchanged. Only spatial dimensions $H$ and $W$ are reduced -- a key distinction from convolutional layers, which can change channel count via number of filters.</span>

---

## <span style="font-size: 16px;">Overlapping vs Non-Overlapping Pooling</span>

### <span style="font-size: 14px;">Non-Overlapping Pooling</span>

<span style="font-size: 14px;">When $s = k$, adjacent windows tile the input without sharing pixels. Common: $k = 2, s = 2$ (halves dimensions). Each input pixel contributes to exactly one output value.</span>

### <span style="font-size: 14px;">Overlapping Pooling</span>

<span style="font-size: 14px;">When $s < k$, adjacent windows share pixels. The overlap is $k - s$ pixels along each axis. AlexNet uses $k = 3, s = 2$, giving 1 pixel of overlap.</span>

<span style="font-size: 14px;">1D example with input $[0, 1, 2, 3, 4]$, $k=3$, $s=2$:</span>

* <span style="font-size: 14px;">**Window 1:** positions $[0, 1, 2]$</span>
* <span style="font-size: 14px;">**Window 2:** positions $[2, 3, 4]$</span>

<span style="font-size: 14px;">Position 2 is shared -- this is the overlapping pixel.</span>

### <span style="font-size: 14px;">Why Overlapping Pooling Helps</span>

<span style="font-size: 14px;">Shared pixels create information redundancy, acting as a mild regularizer that produces a smoother representation and makes it harder to memorize exact spatial configurations.</span>

<span style="font-size: 14px;">It also provides gentler downsampling: $k=3, s=3$ turns 27x27 into 9x9, while $k=3, s=2$ yields 13x13, preserving more spatial resolution.</span>

---

## <span style="font-size: 16px;">Paper Context and Design Decisions</span>

### <span style="font-size: 14px;">What the Paper Says</span>

<span style="font-size: 14px;">Krizhevsky et al. (2012) state: "We generally observe that models with overlapping pooling find it slightly more difficult to overfit." This was significant because overfitting was a major concern for AlexNet's ~60 million parameters trained on only 1.2 million images.</span>

### <span style="font-size: 14px;">Measured Error Reduction</span>

<span style="font-size: 14px;">Switching from non-overlapping ($k=2, s=2$) to overlapping ($k=3, s=2$) reduced top-1 error by 0.4% and top-5 error by 0.3%. These fractions mattered in ILSVRC where top entries were separated by less than 1%.</span>

### <span style="font-size: 14px;">Where Pooling Is Applied in AlexNet</span>

<span style="font-size: 14px;">Pooling is applied selectively, not after every conv layer:</span>

* <span style="font-size: 14px;">**After Conv1:** Pool1 applied</span>
* <span style="font-size: 14px;">**After Conv2:** Pool2 applied</span>
* <span style="font-size: 14px;">**After Conv3:** No pooling</span>
* <span style="font-size: 14px;">**After Conv4:** No pooling</span>
* <span style="font-size: 14px;">**After Conv5:** Pool5 applied</span>

<span style="font-size: 14px;">Pooling is omitted after Conv3/Conv4 because spatial dimensions are already small (13x13); further pooling would shrink feature maps too aggressively.</span>

### <span style="font-size: 14px;">Regularization Role</span>

<span style="font-size: 14px;">Overlapping pooling was one of several regularization techniques in AlexNet, alongside data augmentation (random crops, flips, PCA color augmentation) and dropout in FC layers. Each contributed a small but additive improvement.</span>

---

## <span style="font-size: 16px;">Where Pooling Fits in the AlexNet Pipeline</span>

<span style="font-size: 14px;">Each convolutional block follows: **Conv -> ReLU -> (optional LRN) -> (optional Pool)**. LRN is applied after Conv1 and Conv2; pooling after Conv1, Conv2, and Conv5.</span>

### <span style="font-size: 14px;">Full Spatial Dimension Trace</span>

<span style="font-size: 14px;">All pooling layers use $k = 3, s = 2$.</span>

<span style="font-size: 14px;">**Input:** $224 \times 224 \times 3$</span>

<span style="font-size: 14px;">**Conv1:** 96 filters, $11 \times 11$, stride 4, no padding.</span>

$$
H_{out} = \left\lfloor \frac{224 - 11}{4} \right\rfloor + 1 = 54
$$

<span style="font-size: 14px;">Some implementations use 227x227 to get exactly 55x55. Following paper convention: $55 \times 55 \times 96$.</span>

<span style="font-size: 14px;">**Pool1:** $k = 3, s = 2$</span>

$$
H_{out} = \left\lfloor \frac{55 - 3}{2} \right\rfloor + 1 = 27
$$

<span style="font-size: 14px;">**After Pool1:** $27 \times 27 \times 96$</span>

<span style="font-size: 14px;">**Conv2:** 256 filters, $5 \times 5$, stride 1, padding 2. Output: $27 \times 27 \times 256$</span>

<span style="font-size: 14px;">**Pool2:** $k = 3, s = 2$</span>

$$
H_{out} = \left\lfloor \frac{27 - 3}{2} \right\rfloor + 1 = 13
$$

<span style="font-size: 14px;">**After Pool2:** $13 \times 13 \times 256$</span>

<span style="font-size: 14px;">**Conv3:** 384 filters, $3 \times 3$, stride 1, padding 1. Output: $13 \times 13 \times 384$ (no pooling)</span>

<span style="font-size: 14px;">**Conv4:** 384 filters, $3 \times 3$, stride 1, padding 1. Output: $13 \times 13 \times 384$ (no pooling)</span>

<span style="font-size: 14px;">**Conv5:** 256 filters, $3 \times 3$, stride 1, padding 1. Output: $13 \times 13 \times 256$</span>

<span style="font-size: 14px;">**Pool5:** $k = 3, s = 2$</span>

$$
H_{out} = \left\lfloor \frac{13 - 3}{2} \right\rfloor + 1 = 6
$$

<span style="font-size: 14px;">**After Pool5:** $6 \times 6 \times 256$</span>

<span style="font-size: 14px;">Flattened to $6 \times 6 \times 256 = 9216$, feeding FC6 (4096) -> FC7 (4096) -> FC8 (1000 classes) with softmax.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Overlapping max pooling with $k = 3, s = 2$ on a 5x5 single-channel feature map.</span>

### <span style="font-size: 14px;">Input Feature Map (5x5)</span>

$$
X = \begin{bmatrix} 1 & 3 & 2 & 7 & 4 \\ 5 & 6 & 8 & 1 & 3 \\ 2 & 9 & 4 & 6 & 5 \\ 3 & 1 & 7 & 2 & 8 \\ 4 & 5 & 3 & 9 & 1 \end{bmatrix}
$$

### <span style="font-size: 14px;">Output Dimensions</span>

$$
H_{out} = \left\lfloor \frac{5 - 3}{2} \right\rfloor + 1 = 2
$$

<span style="font-size: 14px;">Output is $2 \times 2$ with 4 pooling windows.</span>

### <span style="font-size: 14px;">Window Positions and Their Contents</span>

<span style="font-size: 14px;">**Window (0, 0):** rows 0-2, columns 0-2</span>

$$
W_{0,0} = \begin{bmatrix} 1 & 3 & 2 \\ 5 & 6 & 8 \\ 2 & 9 & 4 \end{bmatrix}
$$

<span style="font-size: 14px;">$\max(W_{0,0}) = 9$</span>

<span style="font-size: 14px;">**Window (0, 1):** rows 0-2, columns 2-4</span>

$$
W_{0,1} = \begin{bmatrix} 2 & 7 & 4 \\ 8 & 1 & 3 \\ 4 & 6 & 5 \end{bmatrix}
$$

<span style="font-size: 14px;">$\max(W_{0,1}) = 8$</span>

<span style="font-size: 14px;">**Window (1, 0):** rows 2-4, columns 0-2</span>

$$
W_{1,0} = \begin{bmatrix} 2 & 9 & 4 \\ 3 & 1 & 7 \\ 4 & 5 & 3 \end{bmatrix}
$$

<span style="font-size: 14px;">$\max(W_{1,0}) = 9$</span>

<span style="font-size: 14px;">**Window (1, 1):** rows 2-4, columns 2-4</span>

$$
W_{1,1} = \begin{bmatrix} 4 & 6 & 5 \\ 7 & 2 & 8 \\ 3 & 9 & 1 \end{bmatrix}
$$

<span style="font-size: 14px;">$\max(W_{1,1}) = 9$</span>

### <span style="font-size: 14px;">Output Feature Map (2x2)</span>

$$
Y = \begin{bmatrix} 9 & 8 \\ 9 & 9 \end{bmatrix}
$$

### <span style="font-size: 14px;">Identifying the Overlap Regions</span>

* <span style="font-size: 14px;">**Horizontal overlap (0,0)-(0,1):** Column 2 (rows 0-2), values $(0,2)=2$, $(1,2)=8$, $(2,2)=4$ shared.</span>
* <span style="font-size: 14px;">**Vertical overlap (0,0)-(1,0):** Row 2 (cols 0-2), values $(2,0)=2$, $(2,1)=9$, $(2,2)=4$ shared.</span>
* <span style="font-size: 14px;">**Horizontal overlap (1,0)-(1,1):** Column 2 (rows 2-4), values $(2,2)=4$, $(3,2)=7$, $(4,2)=3$ shared.</span>
* <span style="font-size: 14px;">**Vertical overlap (0,1)-(1,1):** Row 2 (cols 2-4), values $(2,2)=4$, $(2,3)=6$, $(2,4)=5$ shared.</span>
* <span style="font-size: 14px;">**Corner overlap:** Pixel $(2,2)=4$ is shared by all four windows.</span>

<span style="font-size: 14px;">With non-overlapping $k = 2, s = 2$, no pixel would appear in more than one window.</span>

---

## <span style="font-size: 16px;">Pooling in Modern Architectures</span>

### <span style="font-size: 14px;">Strided Convolutions Replacing Pooling</span>

<span style="font-size: 14px;">Springenberg et al. (2015) showed max pooling can be replaced by stride-2 convolutions without accuracy loss. ResNet (He et al., 2015) adopted this. The advantage: downsampling becomes learnable rather than fixed.</span>

### <span style="font-size: 14px;">Global Average Pooling</span>

<span style="font-size: 14px;">Lin et al. (2014) introduced GAP as a replacement for FC layers. It computes the spatial average per channel, outputting a vector of length $C$ from a $C \times H \times W$ feature map. This dramatically reduces parameters and was adopted by GoogLeNet (Szegedy et al., 2015) and ResNet.</span>

### <span style="font-size: 14px;">Adaptive Pooling in PyTorch</span>

<span style="font-size: 14px;">PyTorch's `nn.AdaptiveMaxPool2d` and `nn.AdaptiveAvgPool2d` let you specify desired output size rather than kernel/stride. For example, `nn.AdaptiveAvgPool2d((1, 1))` implements GAP regardless of input size.</span>

### <span style="font-size: 14px;">Why Pooling Is Less Common in Modern Designs</span>

* <span style="font-size: 14px;">**Information loss:** Max pooling permanently discards spatial info, harming tasks requiring fine-grained detail (detection, segmentation).</span>
* <span style="font-size: 14px;">**Learnable alternatives:** Strided convolutions achieve the same reduction while learning optimal downsampling via backpropagation.</span>
* <span style="font-size: 14px;">**Architectural simplicity:** Modern architectures favor uniform building blocks.</span>
* <span style="font-size: 14px;">**Attention mechanisms:** Transformer-based vision models use patch merging rather than pooling.</span>

<span style="font-size: 14px;">Despite these trends, GAP before the final classifier remains common, and max pooling persists in lightweight edge models.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

### <span style="font-size: 14px;">Confusing the Pooling Output Formula with the Convolution Output Formula</span>

<span style="font-size: 14px;">AlexNet pooling uses zero padding. The correct formula is:</span>

$$
H_{out} = \left\lfloor \frac{H_{in} - k}{s} \right\rfloor + 1
$$

<span style="font-size: 14px;">Do not add a $+2p$ term as for convolution. This error propagates through the network, causing dimension mismatches.</span>

### <span style="font-size: 14px;">Forgetting That Pooling Preserves the Channel Count</span>

<span style="font-size: 14px;">Pooling operates per channel and does not change channel count. Input $96 \times 27 \times 27$ with $k=3, s=2$ produces $96 \times 13 \times 13$. Unlike convolution, where output channels equal number of filters.</span>

### <span style="font-size: 14px;">Off-by-One Errors with Overlapping Windows</span>

<span style="font-size: 14px;">The floor operation and $+1$ are easy to forget. Example: $H_{in}=13, k=3, s=2$:</span>

$$
H_{out} = \left\lfloor \frac{13 - 3}{2} \right\rfloor + 1 = 6
$$

<span style="font-size: 14px;">Forgetting $+1$ gives 5 instead of 6. Also, overlap width is $k - s$ pixels (1 pixel for $k=3, s=2$), not 2.</span>

### <span style="font-size: 14px;">Assuming Pooling Has Learnable Parameters</span>

<span style="font-size: 14px;">Max and average pooling have zero learnable parameters. When counting AlexNet's ~60M parameters, pooling contributes zero. Behavior is fixed once $k$ and $s$ are chosen.</span>

### <span style="font-size: 14px;">Ignoring That Pooling Discards Spatial Information Permanently</span>

<span style="font-size: 14px;">Max pooling is not invertible -- the locations and values of non-max elements are permanently lost.</span>

* <span style="font-size: 14px;">**For classification:** Acceptable, since final prediction only needs to know what is in the image.</span>
* <span style="font-size: 14px;">**For segmentation/detection:** Problematic. Architectures like U-Net and FPN use skip connections to recover discarded spatial info.</span>
* <span style="font-size: 14px;">**For unpooling:** Requires storing max indices during forward pass, adding memory overhead.</span>

---
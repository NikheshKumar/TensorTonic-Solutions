# <span style="font-size: 20px;">Max Pool 2D</span>

<span style="font-size: 14px;">Max pooling is a parameter-free downsampling operation that slides a window over a feature map and keeps only the maximum value in each window. Introduced in early convolutional networks (LeCun et al., 1998) and central to AlexNet (Krizhevsky et al., 2012) and VGG (Simonyan and Zisserman, 2014), it reduces spatial resolution while preserving the strongest activations, giving the network a measure of local translation invariance.</span>

---

## <span style="font-size: 16px;">What It Does</span>

<span style="font-size: 14px;">A pooling layer partitions the spatial dimensions of a feature map into (possibly overlapping) windows and replaces each window with a single summary statistic. For **max pooling** that statistic is the maximum, which selects the most salient response in each region.</span>

<span style="font-size: 14px;">The operation serves three roles in a convolutional network:</span>

* <span style="font-size: 14px;">**Downsampling:** it shrinks $H$ and $W$, cutting the activation memory and the FLOPs of every subsequent layer. A stride-2 pool quarters the spatial area.</span>
* <span style="font-size: 14px;">**Local translation invariance:** if the strongest feature shifts by a pixel or two within a window, the pooled output is unchanged, so the network becomes less sensitive to exact position.</span>
* <span style="font-size: 14px;">**Receptive-field growth:** by collapsing spatial extent, later layers see a larger fraction of the original image per unit, helping build a hierarchy from edges to objects.</span>

<span style="font-size: 14px;">Crucially, max pooling has **no learnable parameters**. It applies a fixed reduction, so it adds capacity for abstraction without adding weights to train.</span>

---

## <span style="font-size: 16px;">Output Size Formula</span>

<span style="font-size: 14px;">Given an input of height $H$ and width $W$, window size $k$, stride $s$, and padding $p$, the output spatial dimensions are:</span>

$$
H_{out} = \left\lfloor \frac{H + 2p - k}{s} \right\rfloor + 1, \quad W_{out} = \left\lfloor \frac{W + 2p - k}{s} \right\rfloor + 1
$$

<span style="font-size: 14px;">With no padding ($p = 0$) this reduces to the form in the problem statement. The $\lfloor \cdot \rfloor$ floor reflects that a partial window at the right or bottom edge is **dropped**, not zero-padded by default.</span>

<span style="font-size: 14px;">Each term has a clear meaning:</span>

* <span style="font-size: 14px;">$H + 2p$ is the padded height; padding lets windows extend past the original border.</span>
* <span style="font-size: 14px;">Subtracting $k$ accounts for the window itself: the last valid window top-left sits at row $H + 2p - k$.</span>
* <span style="font-size: 14px;">Dividing by $s$ counts how many stride steps fit, and the $+1$ counts the first window at position 0.</span>

<span style="font-size: 14px;">The channel dimension is **untouched**: pooling acts independently per channel, so a $(C, H, W)$ tensor becomes $(C, H_{out}, W_{out})$. This is unlike convolution, which mixes channels.</span>

---

## <span style="font-size: 16px;">Parameters, FLOPs, and Receptive Field</span>

<span style="font-size: 14px;">Max pooling is exceptionally cheap, which is part of why it was attractive in compute-limited early architectures.</span>

* <span style="font-size: 14px;">**Parameters:** zero. There is no weight tensor, no bias, nothing to store or update. The layer is a fixed function of its inputs.</span>
* <span style="font-size: 14px;">**FLOPs:** producing each output element requires $k \cdot k - 1$ comparisons. The total cost is $C \cdot H_{out} \cdot W_{out} \cdot (k^2 - 1)$ comparisons, far below the multiply-accumulate cost of a convolution that mixes $C_{in}$ channels.</span>
* <span style="font-size: 14px;">**Memory:** the forward pass must additionally cache one argmax index per output element so the backward pass can route gradients, costing $C \cdot H_{out} \cdot W_{out}$ integers.</span>

<span style="font-size: 14px;">For the **receptive field**, a single pool of window $k$ and stride $s$ multiplies the stride product of the network (the jump) by $s$ and grows the receptive field by $(k-1)$ times the prior jump. Stacking pools is the primary mechanism by which deep CNNs let a single deep-layer unit "see" a large region of the input image, building from local edges to global shapes.</span>

<span style="font-size: 14px;">Concretely, two $2 \times 2$ stride-2 pools separated by a $3 \times 3$ convolution give a deep unit a receptive field of tens of input pixels even though every individual operation is local. This compounding is why a handful of downsampling stages suffice to take a $224 \times 224$ image down to a $7 \times 7$ feature map in VGG, at which point each location summarizes a substantial image patch.</span>

---

## <span style="font-size: 16px;">The Pooling Operation</span>

<span style="font-size: 14px;">For each output position $(i, j)$, the window's top-left corner in the input is $(i \cdot s, j \cdot s)$, and the output is the maximum over the $k \times k$ block:</span>

$$
\text{out}[i, j] = \max_{0 \le a < k,\; 0 \le b < k} \text{image}[i \cdot s + a,\; j \cdot s + b]
$$

<span style="font-size: 14px;">The algorithm in steps:</span>

<span style="font-size: 14px;">1. **Compute output shape** using the formula above; allocate an $H_{out} \times W_{out}$ result.</span>

<span style="font-size: 14px;">2. **Slide the window** by iterating $i$ from $0$ to $H_{out}-1$ and $j$ from $0$ to $W_{out}-1$. The stride $s$ converts an output index into an input offset.</span>

<span style="font-size: 14px;">3. **Reduce** by scanning the $k \times k$ block and recording its maximum into $\text{out}[i,j]$.</span>

<span style="font-size: 14px;">When $s = k$ (the common case, e.g. $2 \times 2$ stride 2) windows tile the input without overlap. When $s < k$ windows overlap, which AlexNet used ($3 \times 3$ windows, stride 2) and reported as slightly reducing overfitting. When $s > k$ some input pixels fall in no window and never contribute, which is almost always a configuration mistake rather than an intent.</span>

<span style="font-size: 14px;">A subtle but important property: max pooling is a **nonlinear** operation. Unlike average pooling, the maximum cannot be expressed as a linear combination of its inputs, so max pooling cannot be folded into an adjacent convolution. This nonlinearity is part of what makes the operation a feature **selector** rather than a feature **blender**: it propagates only the dominant response and discards the rest, which sharpens edge and texture detectors in the early layers of AlexNet and VGG.</span>

---

## <span style="font-size: 16px;">Backward Pass and the Argmax</span>

<span style="font-size: 14px;">Max pooling is not differentiable in the classical sense, but it has a well-defined subgradient. During the forward pass, each output records **which input element was the maximum** (the argmax). During backpropagation, the incoming gradient is routed entirely to that one position; every other element in the window receives a gradient of zero.</span>

$$
\frac{\partial \, \text{out}[i,j]}{\partial \, \text{image}[a,b]} = \begin{cases} 1 & (a,b) = \arg\max \\ 0 & \text{otherwise} \end{cases}
$$

<span style="font-size: 14px;">This is why frameworks cache the argmax indices in the forward pass. The gradient is **sparse**: only the winning location in each window learns, which is one reason max pooling behaves like a selective feature detector. When windows overlap, a single input position may be the max for several output windows, and the gradients accumulate (sum) at that location.</span>

<span style="font-size: 14px;">There is a consequence for training dynamics worth noting. Because only the argmax receives gradient, units that are never the maximum in any window receive no learning signal through the pool. In practice this is rarely a problem because the argmax shifts as weights change, but it does mean the gradient flowing back through a max pool is much sparser than through an average pool, where every input in the window receives an equal share $1/k^2$ of the upstream gradient.</span>

---

## <span style="font-size: 16px;">Padding, Edges, and Ceiling Mode</span>

<span style="font-size: 14px;">The default convention, and the one this problem uses, applies **no padding** and floors the output size, silently dropping any leftover row or column on the right and bottom edges. Two related options change this behaviour:</span>

* <span style="font-size: 14px;">**Negative-infinity padding:** when padding is requested for max pooling, frameworks pad with $-\infty$ rather than zero. Padding with zero would be wrong, because a zero could become the window maximum and leak a phantom value into the output; $-\infty$ guarantees padded positions never win the max.</span>
* <span style="font-size: 14px;">**Ceiling mode:** an option that replaces the floor with a ceiling in the size formula, keeping a final partial window instead of discarding it. This produces a slightly larger output and includes the edge data, at the cost of windows that extend past the original border.</span>

<span style="font-size: 14px;">These choices matter when matching reference behaviour exactly. A from-scratch implementation that pads max pooling with zeros, or that rounds up instead of down, will diverge from `torch.nn.functional.max_pool2d` on edge cases even when the interior is correct.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Take a $4 \times 4$ input, $k = 2$, $s = 2$, $p = 0$:</span>

$$
\text{image} = \begin{pmatrix} 1 & 3 & 2 & 4 \\ 5 & 6 & 1 & 2 \\ 7 & 2 & 9 & 0 \\ 1 & 8 & 3 & 4 \end{pmatrix}
$$

<span style="font-size: 14px;">Output size: $H_{out} = \lfloor (4 - 2)/2 \rfloor + 1 = 2$, likewise $W_{out} = 2$, so the result is $2 \times 2$.</span>

<span style="font-size: 14px;">1. **Top-left window** (rows 0-1, cols 0-1): values $1, 3, 5, 6$, max is $6$.</span>

<span style="font-size: 14px;">2. **Top-right window** (rows 0-1, cols 2-3): values $2, 4, 1, 2$, max is $4$.</span>

<span style="font-size: 14px;">3. **Bottom-left window** (rows 2-3, cols 0-1): values $7, 2, 1, 8$, max is $8$.</span>

<span style="font-size: 14px;">4. **Bottom-right window** (rows 2-3, cols 2-3): values $9, 0, 3, 4$, max is $9$.</span>

$$
\text{out} = \begin{pmatrix} 6 & 4 \\ 8 & 9 \end{pmatrix}
$$

<span style="font-size: 14px;">The backward pass would send gradients only to positions $(1,1)$, $(0,3)$, $(3,1)$ and $(2,2)$, the four argmax locations.</span>

<span style="font-size: 14px;">To see the translation-invariance claim concretely, suppose the value $6$ at position $(1,1)$ instead appeared at $(0,0)$ within the same top-left window. The window's maximum is still $6$, so the output is unchanged. A small shift of the dominant feature inside a window leaves the pooled response constant; only when the feature crosses a window boundary does the output move. This is the precise sense in which max pooling buys **local** invariance, bounded by the window size.</span>

---

## <span style="font-size: 16px;">Max Pooling Versus Average Pooling</span>

<span style="font-size: 14px;">The two classic pooling operations differ only in the reduction applied, but the behavioural consequences are large:</span>

* <span style="font-size: 14px;">**Max pooling selects** the single strongest activation, acting as a feature detector. It is robust to clutter: a strong edge response survives even if surrounded by weak responses. This suits early layers detecting sparse, high-contrast features.</span>
* <span style="font-size: 14px;">**Average pooling smooths**, mixing every value in the window. It preserves overall intensity and background context but can dilute a sharp feature when most of the window is inactive.</span>
* <span style="font-size: 14px;">**Gradient behaviour differs:** max routes the full gradient to one location, average splits it $1/k^2$ across all locations. Average pooling therefore gives every input a learning signal, while max pooling gives a sparse one.</span>

<span style="font-size: 14px;">Empirically, max pooling tended to win for classification feature extraction in the AlexNet and VGG era, while average pooling found its place at the network head (global average pool) where smoothing the whole feature map into a class-descriptor vector is exactly what is wanted.</span>

---

## <span style="font-size: 16px;">Design Context and Modern Trends</span>

<span style="font-size: 14px;">Max pooling dominated early CNNs because it cheaply builds invariance and reduces resolution. AlexNet and VGG interleave $2 \times 2$ max pools between conv stacks. However, several later designs **reduce or remove** pooling:</span>

* <span style="font-size: 14px;">**Strided convolution as a replacement:** "Striving for Simplicity" (Springenberg et al., 2015) showed that a stride-2 convolution can subsume pooling, downsampling with learnable weights. ResNet uses strided convs for most downsampling.</span>
* <span style="font-size: 14px;">**Global average pooling at the head:** ResNet and GoogLeNet replace large fully connected classifiers with a single global pool, collapsing $H \times W$ to $1 \times 1$.</span>
* <span style="font-size: 14px;">**Max pooling persists** where preserving the strongest activation matters, such as the stem of ResNet (a $3 \times 3$ stride-2 max pool after the first conv).</span>

<span style="font-size: 14px;">A further reason modern transformer-based vision models (ViT, Swin) abandon max pooling is that patch-based tokenization and attention provide downsampling and context aggregation through learned mechanisms, so a fixed max reduction is no longer the natural fit. Even so, max pooling remains a textbook operation and a building block in countless deployed CNNs, and understanding its exact arithmetic and gradient routing is foundational.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Forgetting the floor and dropping a partial window.** With $H = 5$, $k = 2$, $s = 2$, the formula gives $\lfloor 3/2 \rfloor + 1 = 2$, so the last row is discarded. Writers who expect $3$ outputs get a shape mismatch. The default PyTorch behaviour does not pad the leftover, it ignores it.</span>
* <span style="font-size: 14px;">**Confusing stride with kernel size.** When $s \ne k$ windows overlap or skip pixels. Using $s = 1$ with $k = 2$ produces nearly the same spatial size as the input, not a downsample, a common surprise when the stride defaults are misread.</span>
* <span style="font-size: 14px;">**Pooling across channels by mistake.** Pooling is strictly spatial and per-channel. Reducing over the channel axis instead collapses feature identity and destroys the representation; the output channel count must equal the input channel count.</span>
* <span style="font-size: 14px;">**Wrong gradient routing in a from-scratch backward pass.** The gradient must flow only to the cached argmax index, not be spread evenly. Distributing it across the window (treating max like average) silently corrupts training and slows or breaks convergence.</span>

---
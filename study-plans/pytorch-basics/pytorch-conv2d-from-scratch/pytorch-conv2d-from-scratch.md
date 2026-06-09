<span style="font-size: 14px;">Convolution is the fundamental operation behind convolutional neural networks (CNNs). Instead of connecting every input to every output as a fully connected layer does, a convolution uses a small **kernel** (also called a filter) that slides across the spatial dimensions of the input, computing a weighted sum at each position. This dramatically reduces the number of parameters and exploits the spatial structure of data like images.</span>

## <span style="font-size: 14px;">From Fully Connected to Convolution</span>

<span style="font-size: 14px;">Consider a grayscale image of size</span> $H \times W$<span style="font-size: 14px;">. A fully connected layer mapping this to an output of the same size would need</span> $(H \times W)^2$ <span style="font-size: 14px;">parameters. For a modest 28x28 image, that is over 600,000 weights for a single layer. A convolution with a 3x3 kernel needs just 9 weights (plus a bias), regardless of the image size. This works because the same kernel is applied at every spatial position: a property called **weight sharing**.</span>

<span style="font-size: 14px;">Weight sharing also introduces **translation equivariance**: if the input shifts, the output shifts by the same amount. This is exactly the right inductive bias for visual data, where a feature (like an edge) should be detected the same way regardless of where it appears.</span>

## <span style="font-size: 14px;">The Convolution Operation</span>

<span style="font-size: 14px;">For a single input channel and a single output channel, 2D convolution works as follows. Given:</span>

* <span style="font-size: 14px;">Input tensor</span> $x$ <span style="font-size: 14px;">of shape</span> $(H, W)$
* <span style="font-size: 14px;">Kernel</span> $w$ <span style="font-size: 14px;">of shape</span> $(k, k)$

<span style="font-size: 14px;">The output at position</span> $(i, j)$ <span style="font-size: 14px;">is:</span>

$$
y_{i,j} = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} x_{i+m,\, j+n} \cdot w_{m,n} + b
$$

<span style="font-size: 14px;">where</span> $b$ <span style="font-size: 14px;">is a scalar bias. The kernel slides across all valid positions, producing an output of shape:</span>

$$
H_{\text{out}} = H - k + 1, \quad W_{\text{out}} = W - k + 1
$$

<span style="font-size: 14px;">This is called "valid" convolution because the kernel only visits positions where it fits entirely within the input. No padding is added.</span>

## <span style="font-size: 14px;">Multiple Channels</span>

<span style="font-size: 14px;">Real inputs typically have multiple channels. A color image has 3 channels (RGB), and intermediate layers in a CNN can have hundreds. The convolution generalizes naturally:</span>

* <span style="font-size: 14px;">Input has</span> $C_{\text{in}}$ <span style="font-size: 14px;">channels, so</span> $x$ <span style="font-size: 14px;">has shape</span> $(C_{\text{in}}, H, W)$
* <span style="font-size: 14px;">Each output channel has its own kernel of shape</span> $(C_{\text{in}}, k, k)$<span style="font-size: 14px;">, which spans all input channels</span>
* <span style="font-size: 14px;">With</span> $C_{\text{out}}$ <span style="font-size: 14px;">output channels, the full weight tensor has shape</span> $(C_{\text{out}}, C_{\text{in}}, k, k)$
* <span style="font-size: 14px;">Each output channel also has its own scalar bias, giving a bias vector of shape</span> $(C_{\text{out}},)$

<span style="font-size: 14px;">The output at channel</span> $c$ <span style="font-size: 14px;">and position</span> $(i, j)$ <span style="font-size: 14px;">is:</span>

$$
\begin{aligned}
y_{c,i,j} = \sum_{c'=0}^{C_{\text{in}}-1}
&\sum_{m=0}^{k-1} \sum_{n=0}^{k-1} \\
&x_{c',\, i+m,\, j+n} \cdot w_{c,c',m,n} + b_c
\end{aligned}
$$

<span style="font-size: 14px;">Each output channel looks at **all** input channels through its own set of weights. This is what allows the network to learn complex cross-channel features.</span>

## <span style="font-size: 14px;">Batched Inputs</span>

<span style="font-size: 14px;">In practice, inputs come in batches. The full input tensor has shape</span> $(N, C_{\text{in}}, H, W)$ <span style="font-size: 14px;">where</span> $N$ <span style="font-size: 14px;">is the batch size. The convolution applies independently to each sample in the batch, producing output of shape</span> $(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})$<span style="font-size: 14px;">. The weights are shared across all samples.</span>

## <span style="font-size: 14px;">The Sliding Window Viewpoint</span>

<span style="font-size: 14px;">The most intuitive way to implement convolution is with a sliding window:</span>

* <span style="font-size: 14px;">For each output position</span> $(i, j)$<span style="font-size: 14px;">, extract the patch</span> $x[:, :, i:i+k, j:j+k]$ <span style="font-size: 14px;">of shape</span> $(N, C_{\text{in}}, k, k)$
* <span style="font-size: 14px;">Flatten the patch to shape</span> $(N, C_{\text{in}} \cdot k \cdot k)$
* <span style="font-size: 14px;">Flatten the weight to shape</span> $(C_{\text{out}}, C_{\text{in}} \cdot k \cdot k)$
* <span style="font-size: 14px;">Compute</span> $\text{output}[:, :, i, j] = \text{patch\_flat} \times \text{weight\_flat}^T + b$ <span style="font-size: 14px;">using matrix multiplication</span>

<span style="font-size: 14px;">This approach loops over spatial positions but vectorizes across the batch and channel dimensions. For small kernels and inputs it is straightforward and educational.</span>

## <span style="font-size: 14px;">The im2col Approach</span>

<span style="font-size: 14px;">Production implementations avoid the spatial loop entirely using a technique called **im2col** (image to column). The idea is to extract all patches at once and stack them into a single large matrix:</span>

* <span style="font-size: 14px;">Extract all</span> $H_{\text{out}} \times W_{\text{out}}$ <span style="font-size: 14px;">patches from the input</span>
* <span style="font-size: 14px;">Reshape each patch from</span> $(C_{\text{in}}, k, k)$ <span style="font-size: 14px;">to a column of length</span> $C_{\text{in}} \cdot k^2$
* <span style="font-size: 14px;">Stack all columns into a matrix of shape</span> $(C_{\text{in}} \cdot k^2,\ H_{\text{out}} \cdot W_{\text{out}})$

<span style="font-size: 14px;">Then the entire convolution becomes a single matrix multiplication:</span>

$$
Y_{\text{flat}} = W_{\text{flat}} \cdot X_{\text{col}} + b
$$

<span style="font-size: 14px;">where</span> $W_{\text{flat}}$ <span style="font-size: 14px;">has shape</span> $(C_{\text{out}}, C_{\text{in}} \cdot k^2)$ <span style="font-size: 14px;">and</span> $X_{\text{col}}$ <span style="font-size: 14px;">has shape</span> $(C_{\text{in}} \cdot k^2, H_{\text{out}} \cdot W_{\text{out}})$<span style="font-size: 14px;">. The output</span> $Y_{\text{flat}}$ <span style="font-size: 14px;">has shape</span> $(C_{\text{out}}, H_{\text{out}} \cdot W_{\text{out}})$ <span style="font-size: 14px;">and is reshaped back to</span> $(C_{\text{out}}, H_{\text{out}}, W_{\text{out}})$<span style="font-size: 14px;">.</span>

<span style="font-size: 14px;">This trades memory for speed: the column matrix duplicates input data (overlapping patches share elements), but the operation maps directly to optimized BLAS routines.</span>

## <span style="font-size: 14px;">Learnable Parameters</span>

<span style="font-size: 14px;">In a convolution layer, the kernel weights and biases are learnable. In PyTorch, this means wrapping them in</span> <code>nn.Parameter</code><span style="font-size: 14px;">:</span>

* <code>self.weight</code> <span style="font-size: 14px;">of shape</span> $(C_{\text{out}}, C_{\text{in}}, k, k)$<span style="font-size: 14px;">, initialized randomly</span>
* <code>self.bias</code> <span style="font-size: 14px;">of shape</span> $(C_{\text{out}},)$<span style="font-size: 14px;">, often initialized to zero</span>

<span style="font-size: 14px;">Using</span> <code>nn.Parameter</code> <span style="font-size: 14px;">tells PyTorch that these tensors should be updated by the optimizer during training. They also appear when you call</span> <code>model.parameters()</code> <span style="font-size: 14px;">or print the model.</span>

## <span style="font-size: 14px;">Output Dimensions</span>

<span style="font-size: 14px;">For a convolution without padding or stride (the simplest case), the output spatial dimensions are:</span>

$$
H_{\text{out}} = H - k + 1
$$

$$
W_{\text{out}} = W - k + 1
$$

<span style="font-size: 14px;">With padding</span> $p$ <span style="font-size: 14px;">and stride</span> $s$<span style="font-size: 14px;">, the general formula is:</span>

$$
H_{\text{out}} = \left\lfloor \frac{H + 2p - k}{s} \right\rfloor + 1
$$

<span style="font-size: 14px;">For this problem, we use no padding (</span>$p = 0$<span style="font-size: 14px;">) and unit stride (</span>$s = 1$<span style="font-size: 14px;">), so only the first formula applies.</span>

## <span style="font-size: 14px;">What Each Output Channel Learns</span>

<span style="font-size: 14px;">Each output channel can be thought of as a different feature detector. In the first layer of a CNN:</span>

* <span style="font-size: 14px;">Some channels might detect horizontal edges</span>
* <span style="font-size: 14px;">Some might detect vertical edges</span>
* <span style="font-size: 14px;">Some might detect corners or textures</span>

<span style="font-size: 14px;">In deeper layers, the features become more abstract: parts of objects, shapes, semantic patterns. Each output channel has its own kernel that spans all input channels, allowing it to combine low-level features into higher-level ones.</span>

## <span style="font-size: 14px;">Receptive Field</span>

<span style="font-size: 14px;">The **receptive field** of an output neuron is the region of the original input that influences its value. For a single convolution layer with kernel size</span> $k$<span style="font-size: 14px;">, each output neuron sees a</span> $k \times k$ <span style="font-size: 14px;">patch. Stacking multiple convolution layers increases the receptive field: two 3x3 layers give an effective receptive field of 5x5, three give 7x7. This is why deep CNNs can capture large-scale patterns despite using small kernels.</span>

## <span style="font-size: 14px;">Convolution vs Cross-Correlation</span>

<span style="font-size: 14px;">Technically, what deep learning frameworks call "convolution" is actually **cross-correlation**. True convolution flips the kernel before sliding:</span>

$$
y_{i,j} = \sum_{m} \sum_{n} x_{i+m,\, j+n} \cdot w_{k-1-m,\, k-1-n}
$$

<span style="font-size: 14px;">In practice, since the kernel weights are learned, flipping makes no difference: the network simply learns the flipped version. All major frameworks (PyTorch, TensorFlow, JAX) use cross-correlation and call it "convolution." This is a universal convention.</span>

## <span style="font-size: 14px;">Parameter Count</span>

<span style="font-size: 14px;">The total number of learnable parameters in a convolution layer is:</span>

$$
\text{params} = C_{\text{out}} \times C_{\text{in}} \times k^2 + C_{\text{out}}
$$

<span style="font-size: 14px;">The first term is the weights, the second is the biases. For example, a layer with 3 input channels, 16 output channels, and a 3x3 kernel has</span> $16 \times 3 \times 9 + 16 = 448 + 16 = 464$ <span style="font-size: 14px;">parameters. Compare this to a fully connected layer between the same flattened input and output, which would need millions of parameters.</span>

## <span style="font-size: 14px;">Implementing as nn.Module</span>

<span style="font-size: 14px;">To implement a convolution layer as an</span> <code>nn.Module</code><span style="font-size: 14px;">:</span>

* <span style="font-size: 14px;">In</span> <code>__init__</code><span style="font-size: 14px;">, create</span> <code>self.weight</code> <span style="font-size: 14px;">and</span> <code>self.bias</code> <span style="font-size: 14px;">as</span> <code>nn.Parameter</code> <span style="font-size: 14px;">with the correct shapes</span>
* <span style="font-size: 14px;">In</span> <code>forward</code><span style="font-size: 14px;">, implement the sliding window operation</span>
* <span style="font-size: 14px;">The weight shape must be</span> $(C_{\text{out}}, C_{\text{in}}, k, k)$ <span style="font-size: 14px;">following PyTorch's convention</span>
* <span style="font-size: 14px;">The bias shape must be</span> $(C_{\text{out}},)$

<span style="font-size: 14px;">The forward pass extracts patches from the input at each spatial position, flattens them, and computes a matrix multiplication with the flattened weights. This is the core of what happens inside</span> <code>torch.nn.Conv2d</code><span style="font-size: 14px;">.</span>

## <span style="font-size: 14px;">Comparison with PyTorch Built-in</span>

<span style="font-size: 14px;">PyTorch's</span> <code>nn.Conv2d</code> <span style="font-size: 14px;">supports many additional features beyond the basic operation:</span>

* <span style="font-size: 14px;">**Padding**: adds zeros (or other values) around the input to control output size</span>
* <span style="font-size: 14px;">**Stride**: skips positions to downsample the output</span>
* <span style="font-size: 14px;">**Dilation**: spaces out kernel elements to increase receptive field without adding parameters</span>
* <span style="font-size: 14px;">**Groups**: splits channels into independent groups for efficiency (depthwise convolution uses groups = in_channels)</span>
* <span style="font-size: 14px;">**No bias**: option to remove the bias term</span>

<span style="font-size: 14px;">The from-scratch version in this problem covers the simplest case: no padding, stride 1, no dilation, no groups, with bias. This captures the essence of convolution while keeping the implementation focused.</span>

## <span style="font-size: 14px;">Role in CNN Architectures</span>

<span style="font-size: 14px;">Convolution layers are the building blocks of virtually all image-processing neural networks:</span>

* <span style="font-size: 14px;">**LeNet** (1998): The original CNN, used 5x5 convolutions for digit recognition</span>
* <span style="font-size: 14px;">**AlexNet** (2012): Popularized deep CNNs with 11x11, 5x5, and 3x3 kernels</span>
* <span style="font-size: 14px;">**VGGNet** (2014): Showed that stacking many 3x3 convolutions is more effective than using large kernels</span>
* <span style="font-size: 14px;">**ResNet** (2015): Added skip connections between convolution blocks, enabling much deeper networks</span>
* <span style="font-size: 14px;">**EfficientNet** (2019): Systematically scaled width, depth, and resolution of convolution networks</span>

<span style="font-size: 14px;">Even in the era of transformers, convolutions remain essential in vision tasks (ConvNeXt, hybrid architectures) and appear in speech, audio, and time-series models.</span>

## <span style="font-size: 14px;">Computational Complexity</span>

<span style="font-size: 14px;">The number of multiply-add operations for a single convolution layer is:</span>

$$
\text{FLOPs} = C_{\text{out}} \times C_{\text{in}} \times k^2 \times H_{\text{out}} \times W_{\text{out}}
$$

<span style="font-size: 14px;">This grows with both the number of channels and the spatial resolution. Techniques to reduce this cost include:</span>

* <span style="font-size: 14px;">**Depthwise separable convolution**: splits into a per-channel spatial convolution followed by a 1x1 pointwise convolution, reducing FLOPs by a factor of roughly</span> $k^2$
* <span style="font-size: 14px;">**Strided convolution**: reduces spatial resolution, cutting FLOPs proportionally</span>
* <span style="font-size: 14px;">**1x1 convolutions (pointwise)**: mix channels without spatial computation, used as bottleneck layers</span>
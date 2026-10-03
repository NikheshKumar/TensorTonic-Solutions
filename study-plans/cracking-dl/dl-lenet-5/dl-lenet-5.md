# <span style="font-size: 20px;">LeNet-5</span>

## <span style="font-size: 16px;">Historical Significance</span>

<span style="font-size: 14px;">LeNet-5 (LeCun, Bottou, Bengio, Haffner, 1998) was the first convolutional neural network to achieve commercial success, deployed by the US Postal Service for zip code recognition. It established the fundamental CNN design pattern that every modern architecture follows: convolutional feature extraction followed by fully-connected classification.</span>

<span style="font-size: 14px;">Key innovations introduced by LeNet-5:</span>
- <span style="font-size: 14px;">**Weight sharing**: the same filter is applied at every spatial position, dramatically reducing parameters compared to fully-connected alternatives</span>
- <span style="font-size: 14px;">**Spatial hierarchy**: stacking conv + pool layers creates a feature hierarchy from edges to shapes to objects</span>
- <span style="font-size: 14px;">**Subsampling (pooling)**: reducing spatial dimensions between conv layers for translation invariance and computational efficiency</span>
- <span style="font-size: 14px;">**End-to-end training**: the entire network (conv + FC) is trained jointly with backpropagation</span>

## <span style="font-size: 16px;">Architecture Details</span>

<span style="font-size: 14px;">The full architecture for 32x32 grayscale input:</span>

<span style="font-size: 14px;">**Block 1:**</span>
- <span style="font-size: 14px;">Conv2d: 1 input channel, 6 output channels, 5x5 kernel, no padding</span>
- <span style="font-size: 14px;">Spatial: $32 \to 28$ (lose 4 pixels from 5x5 valid conv)</span>
- <span style="font-size: 14px;">ReLU activation</span>
- <span style="font-size: 14px;">AvgPool2d: 2x2 window, stride 2. Spatial: $28 \to 14$</span>

<span style="font-size: 14px;">**Block 2:**</span>
- <span style="font-size: 14px;">Conv2d: 6 input channels, 16 output channels, 5x5 kernel, no padding</span>
- <span style="font-size: 14px;">Spatial: $14 \to 10$</span>
- <span style="font-size: 14px;">ReLU activation</span>
- <span style="font-size: 14px;">AvgPool2d: 2x2 window, stride 2. Spatial: $10 \to 5$</span>

<span style="font-size: 14px;">**Classifier:**</span>
- <span style="font-size: 14px;">Flatten: $16 \times 5 \times 5 = 400$ features</span>
- <span style="font-size: 14px;">Linear: $400 \to 120$ + ReLU</span>
- <span style="font-size: 14px;">Linear: $120 \to 84$ + ReLU</span>
- <span style="font-size: 14px;">Linear: $84 \to C$ (no activation, raw logits)</span>

## <span style="font-size: 16px;">Parameter Count</span>

<span style="font-size: 14px;">Understanding parameter counts is a common interview question:</span>

$$
\begin{aligned}
\text{Conv1:} & \quad 1 \times 6 \times 5 \times 5 + 6 = 156 \\
\text{Conv2:} & \quad 6 \times 16 \times 5 \times 5 + 16 = 2{,}416 \\
\text{FC1:} & \quad 400 \times 120 + 120 = 48{,}120 \\
\text{FC2:} & \quad 120 \times 84 + 84 = 10{,}164 \\
\text{FC3:} & \quad 84 \times 10 + 10 = 850 \\
\text{Total:} & \quad 61{,}706
\end{aligned}
$$

<span style="font-size: 14px;">Notice that the FC layers dominate: 59,134 out of 61,706 parameters (96%) are in the classifier. This motivated the shift toward global average pooling in modern architectures, which eliminates the large FC layers entirely.</span>

## <span style="font-size: 16px;">PyTorch nn.Module Pattern</span>

<span style="font-size: 14px;">LeNet-5 is the canonical example for learning the PyTorch model-building pattern:</span>

- <span style="font-size: 14px;">**Subclass nn.Module**: inherit from `nn.Module` and call `super().__init__()`</span>
- <span style="font-size: 14px;">**Define layers in __init__**: store each layer as a `self.layer_name` attribute. PyTorch automatically tracks parameters of all `nn.Module` attributes.</span>
- <span style="font-size: 14px;">**Implement forward()**: define the computation graph by chaining layer calls. PyTorch's autograd records operations for automatic backward pass computation.</span>
- <span style="font-size: 14px;">**No need to implement backward()**: autograd handles gradient computation automatically based on the operations in `forward()`.</span>

<span style="font-size: 14px;">Common mistakes to avoid:</span>
- <span style="font-size: 14px;">Forgetting `super().__init__()`: parameters will not be registered</span>
- <span style="font-size: 14px;">Using bare Python functions instead of `nn` modules: functional operations work but are harder to inspect and serialize</span>
- <span style="font-size: 14px;">Applying activation after the final layer: cross-entropy loss expects raw logits</span>

## <span style="font-size: 16px;">Original vs Modern LeNet-5</span>

<span style="font-size: 14px;">The original 1998 paper differs from the modern implementation in several ways:</span>
- <span style="font-size: 14px;">**Activations**: original used sigmoid/tanh, modern uses ReLU (faster training, avoids vanishing gradients)</span>
- <span style="font-size: 14px;">**Pooling**: original used a learned subsampling layer, modern uses average or max pooling</span>
- <span style="font-size: 14px;">**Conv2 connectivity**: the original had a complex sparse connectivity pattern between Conv1 and Conv2 channels (only certain input channels connected to certain output channels). Modern implementations use full connectivity.</span>
- <span style="font-size: 14px;">**Output layer**: original used Euclidean RBF units, modern uses softmax cross-entropy</span>

<span style="font-size: 14px;">For interviews, the modern version (ReLU, AvgPool/MaxPool, full connectivity, cross-entropy) is the expected implementation.</span>

## <span style="font-size: 16px;">From LeNet to Modern CNNs</span>

<span style="font-size: 14px;">LeNet established the template that all subsequent CNNs follow. The progression of key architectures:</span>
- <span style="font-size: 14px;">**LeNet-5** (1998): 5 layers, 60K params, proved CNNs work</span>
- <span style="font-size: 14px;">**AlexNet** (2012): 8 layers, 60M params, won ImageNet with ReLU + dropout + GPU training</span>
- <span style="font-size: 14px;">**VGG** (2014): 16-19 layers, 138M params, showed deeper is better with uniform 3x3 convs</span>
- <span style="font-size: 14px;">**GoogLeNet/Inception** (2014): 22 layers, 6.8M params, multi-scale parallel convolutions</span>
- <span style="font-size: 14px;">**ResNet** (2015): 50-152 layers, skip connections enabled extreme depth</span>
- <span style="font-size: 14px;">**EfficientNet** (2019): compound scaling of depth, width, and resolution</span>

<span style="font-size: 14px;">Each architecture builds on LeNet's core insight: hierarchical feature extraction through stacked convolutions.</span>

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

- <span style="font-size: 14px;">**Q: Why does LeNet use 5x5 kernels instead of 3x3?**</span>
  <span style="font-size: 14px;">A: At the time, 5x5 was a practical choice for 32x32 inputs. Modern architectures favor 3x3 kernels because two stacked 3x3 layers have the same receptive field as one 5x5 but with fewer parameters (18 vs 25) and an extra non-linearity.</span>

- <span style="font-size: 14px;">**Q: Why are 96% of parameters in the FC layers?**</span>
  <span style="font-size: 14px;">A: The flatten operation creates a 400-dimensional vector, and the FC1 layer alone needs 400 x 120 = 48,000 weights. Convolution layers share weights across spatial positions, making them parameter-efficient. This problem motivated global average pooling (GoogLeNet, ResNet) which replaces FC layers entirely.</span>

- <span style="font-size: 14px;">**Q: What would change if the input were 28x28 (standard MNIST) instead of 32x32?**</span>
  <span style="font-size: 14px;">A: The spatial dimensions after Conv2 + Pool2 would be: $28 \to 24 \to 12 \to 8 \to 4$. The flatten size becomes $16 \times 4 \times 4 = 256$ instead of 400, requiring FC1 to be $256 \to 120$. Alternatively, you can pad the input to 32x32.</span>

- <span style="font-size: 14px;">**Q: How does model.parameters() know about all the weights?**</span>
  <span style="font-size: 14px;">A: PyTorch's nn.Module uses Python's `__setattr__` to intercept attribute assignments. When you write `self.conv1 = nn.Conv2d(...)`, the module registers conv1 as a submodule and tracks its parameters recursively. This is why you must call `super().__init__()` first.</span>

- <span style="font-size: 14px;">**Q: Should you apply softmax in forward()?**</span>
  <span style="font-size: 14px;">A: No. PyTorch's `nn.CrossEntropyLoss` combines `log_softmax` and `nll_loss` internally, which is numerically more stable than applying softmax separately. The model should output raw logits. Apply softmax only at inference time if you need probabilities.</span>

---
# <span style="font-size: 20px;">Classifier Head</span>

<span style="font-size: 14px;">The classifier head is the final stage of VGGNet (Simonyan and Zisserman, 2014) that transforms extracted feature maps into class predictions. It consists of a flatten operation followed by three fully connected (FC) layers: FC1 with ReLU (4096 units), FC2 with ReLU (4096 units), and FC3 (num_classes units) with no activation. These FC layers contain the vast majority of VGGNet's parameters and represent a critical design decision in early CNN architectures.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">The classifier head sits after VGGNet's convolutional feature extractor. While the convolutional layers learn hierarchical spatial features (edges, textures, object parts), the classifier head maps those features to class scores. It is a multilayer perceptron (MLP) stacked on top of convolutional layers.</span>

<span style="font-size: 14px;">The pipeline is straightforward: the 3D feature volume from the last max-pooling layer is flattened into a 1D vector, then passed through three fully connected layers. The first two use ReLU activation for nonlinearity. The third produces raw output scores (logits) with no activation -- one score per class.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">**Flatten** -- convert the 3D feature volume into a 1D vector:</span>

$$
\mathbf{x} = \text{flatten}(\mathbf{F}) \in \mathbb{R}^{d_{\text{in}}}
$$

<span style="font-size: 14px;">where $\mathbf{F} \in \mathbb{R}^{H \times W \times C}$ is the feature volume and $d_{\text{in}} = H \times W \times C$.</span>

<span style="font-size: 14px;">**FC1 (with ReLU):**</span>

$$
\mathbf{h}_1 = \text{ReLU}(\mathbf{x} W_1 + \mathbf{b}_1)
$$

<span style="font-size: 14px;">where $W_1 \in \mathbb{R}^{d_{\text{in}} \times 4096}$, $\mathbf{b}_1 \in \mathbb{R}^{4096}$.</span>

<span style="font-size: 14px;">**FC2 (with ReLU):**</span>

$$
\mathbf{h}_2 = \text{ReLU}(\mathbf{h}_1 W_2 + \mathbf{b}_2)
$$

<span style="font-size: 14px;">where $W_2 \in \mathbb{R}^{4096 \times 4096}$, $\mathbf{b}_2 \in \mathbb{R}^{4096}$.</span>

<span style="font-size: 14px;">**FC3 (no activation -- raw logits):**</span>

$$
\mathbf{y} = \mathbf{h}_2 W_3 + \mathbf{b}_3
$$

<span style="font-size: 14px;">where $W_3 \in \mathbb{R}^{4096 \times K}$, $\mathbf{b}_3 \in \mathbb{R}^{K}$, and $K$ is the number of classes. The general formula for any fully connected layer is $\mathbf{y} = \mathbf{x} W + \mathbf{b}$ -- every input neuron connects to every output neuron, hence "fully connected."</span>

---

## <span style="font-size: 16px;">Why Flatten</span>

<span style="font-size: 14px;">Convolutional layers produce 3D feature volumes with height, width, and channel dimensions. In VGG16, the final max-pooling layer outputs a tensor of shape $7 \times 7 \times 512$. Fully connected layers operate on 1D vectors, so the 3D volume must be reshaped:</span>

$$
7 \times 7 \times 512 = 25{,}088
$$

<span style="font-size: 14px;">After flattening, spatial structure is discarded -- the FC layer treats all 25,088 values as independent inputs with no notion of which values were spatial neighbors.</span>

<span style="font-size: 14px;">This is deliberate. By the time features reach the classifier head, the convolutional layers have already extracted high-level semantic features. The FC layers need to combine these features globally (e.g., "there is an eye here AND a nose there, so this is a face") rather than locally. Flattening enables this global combination by letting every FC neuron see every feature from every spatial location.</span>

---

## <span style="font-size: 16px;">The Three FC Layers</span>

### <span style="font-size: 14px;">FC1: 25,088 to 4,096</span>

<span style="font-size: 14px;">The first FC layer is the largest in VGGNet by parameter count. It projects the 25,088-dimensional flattened vector down to 4,096 dimensions, containing $25{,}088 \times 4{,}096 + 4{,}096 = 102{,}764{,}544$ parameters (approximately 102.8 million). ReLU is applied after the affine transformation. FC1 serves as a massive dimensionality reduction, compressing the spatially-aware feature representation into a compact vector for classification.</span>

### <span style="font-size: 14px;">FC2: 4,096 to 4,096</span>

<span style="font-size: 14px;">The second FC layer maintains dimensionality: 4,096 in, 4,096 out, with $4{,}096 \times 4{,}096 + 4{,}096 = 16{,}781{,}312$ parameters (approximately 16.8 million). ReLU is applied after the affine transformation. While FC1 compresses spatial features into a global representation, FC2 refines that representation by learning higher-order combinations. Two stacked ReLU-activated FC layers provide sufficient capacity for complex, nonlinear decision boundaries.</span>

### <span style="font-size: 14px;">FC3: 4,096 to num_classes</span>

<span style="font-size: 14px;">The final FC layer projects from 4,096 to $K$ dimensions (for ImageNet, $K = 1{,}000$), with $4{,}096 \times 1{,}000 + 1{,}000 = 4{,}097{,}000$ parameters. No activation function is applied -- the output is raw logits. Each of the $K$ output neurons produces a single score for one class. These scores are not probabilities; they can be any real number, positive or negative.</span>

### <span style="font-size: 14px;">Why 4,096?</span>

<span style="font-size: 14px;">The choice of 4,096 hidden units was inherited from AlexNet. This number is a power of two ($2^{12} = 4{,}096$), which aligns well with GPU memory architectures. The VGGNet paper provides no specific justification -- it was an empirical choice that gave sufficient capacity for ImageNet classification without excessive overfitting (when combined with dropout).</span>

---

## <span style="font-size: 16px;">Why No Activation on FC3</span>

<span style="font-size: 14px;">The final layer produces raw logits with no activation. This is deliberate, not an oversight:</span>

<span style="font-size: 14px;">**Cross-entropy loss expects logits.** PyTorch's `nn.CrossEntropyLoss` internally applies log-softmax before computing negative log-likelihood. Passing pre-softmaxed values applies softmax twice, producing incorrect gradients.</span>

<span style="font-size: 14px;">**Numerical stability.** Computing softmax then log separately can cause issues -- when one logit dominates, softmax produces values near 0 for other classes, and $\log(0)$ is negative infinity. The fused log-softmax avoids this via the log-sum-exp trick, which requires raw logits.</span>

<span style="font-size: 14px;">**Flexibility.** Raw logits work with different post-processing: sigmoid for multi-label classification, temperature scaling for calibration, soft targets for knowledge distillation. Keeping FC3 activation-free preserves all these options.</span>

---

## <span style="font-size: 16px;">Parameter Dominance</span>

<span style="font-size: 14px;">The classifier head contains the vast majority of VGG16's parameters:</span>

* <span style="font-size: 14px;">**Convolutional layers:** ~14.7 million parameters</span>
* <span style="font-size: 14px;">**FC1 (25,088 to 4,096):** ~102.8 million parameters</span>
* <span style="font-size: 14px;">**FC2 (4,096 to 4,096):** ~16.8 million parameters</span>
* <span style="font-size: 14px;">**FC3 (4,096 to 1,000):** ~4.1 million parameters</span>
* <span style="font-size: 14px;">**Total FC:** ~123.6 million of ~138.4 million total</span>

<span style="font-size: 14px;">The three FC layers account for roughly **89%** of VGG16's total parameters, with FC1 alone responsible for about **74%** of the entire network. The convolutional layers that perform the actual feature extraction use only 11% of the parameters.</span>

<span style="font-size: 14px;">This concentration is a direct consequence of the flatten operation. The input dimension of 25,088 multiplied by 4,096 outputs creates an enormous weight matrix with no parameter sharing. Compare this to a $3 \times 3$ conv layer with 512 input and 512 output channels: only $3 \times 3 \times 512 \times 512 = 2{,}359{,}296$ parameters, shared across all spatial positions.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">In "Very Deep Convolutional Networks for Large-Scale Image Recognition" (Simonyan and Zisserman, 2014), the classifier head is described concisely: "The fully-connected layers consist of 4096-4096-1000 channels." This pattern was borrowed directly from AlexNet and retained without modification.</span>

<span style="font-size: 14px;">The paper's primary contribution was demonstrating that network depth with small $3 \times 3$ filters was critical for performance. The classifier head was not a focus of innovation -- it was kept identical to AlexNet's to isolate the effect of convolutional depth.</span>

<span style="font-size: 14px;">**Dropout between FC layers.** The paper specifies "dropout regularisation (dropout ratio set to 0.5)" on the first two FC layers. Dropout randomly zeroes out 50% of neurons during each training pass, forcing redundant representations and preventing co-adaptation. It is applied only during training, only after ReLU in FC1 and FC2, never after FC3.</span>

<span style="font-size: 14px;">**Dense evaluation.** At test time, the FC layers can be converted to equivalent convolutions: FC1 becomes a $7 \times 7$ conv with 4,096 channels, FC2 becomes $1 \times 1$ with 4,096 channels, FC3 becomes $1 \times 1$ with 1,000 channels. This allows processing images of arbitrary size and producing spatially-varying score maps that are averaged for the final prediction.</span>

<span style="font-size: 14px;">**Weight initialization.** Training very deep networks was difficult due to unstable gradients. The authors first trained VGG11 (configuration A) and used its learned FC weights to initialize deeper configurations, bootstrapping the classifier head from the shallow network.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Tracing dimensions through the VGG16 classifier head for a single image:</span>

<span style="font-size: 14px;">**Input:** Last max-pooling outputs shape $(7, 7, 512)$.</span>

<span style="font-size: 14px;">**Step 1 -- Flatten:**</span>

$$
(7, 7, 512) \rightarrow (25{,}088)
$$

<span style="font-size: 14px;">**Step 2 -- FC1 + ReLU:** $\mathbf{x} \in \mathbb{R}^{25088}$, $W_1 \in \mathbb{R}^{25088 \times 4096}$, $\mathbf{b}_1 \in \mathbb{R}^{4096}$. Matrix multiply, add bias, apply ReLU. Pre-ReLU values $[2.3, -0.7, 1.1]$ become $[2.3, 0.0, 1.1]$. Output: $\mathbf{h}_1 \in \mathbb{R}^{4096}$.</span>

<span style="font-size: 14px;">**Step 3 -- FC2 + ReLU:** $\mathbf{h}_1 \in \mathbb{R}^{4096}$, $W_2 \in \mathbb{R}^{4096 \times 4096}$, $\mathbf{b}_2 \in \mathbb{R}^{4096}$. Same computation. Output: $\mathbf{h}_2 \in \mathbb{R}^{4096}$.</span>

<span style="font-size: 14px;">**Step 4 -- FC3:** $\mathbf{h}_2 \in \mathbb{R}^{4096}$, $W_3 \in \mathbb{R}^{4096 \times 1000}$, $\mathbf{b}_3 \in \mathbb{R}^{1000}$. Output: $\mathbf{y} \in \mathbb{R}^{1000}$ -- one raw logit per class. No ReLU, no softmax.</span>

<span style="font-size: 14px;">**Dimension summary:**</span>

$$
(7, 7, 512) \xrightarrow{\text{flatten}} (25088) \xrightarrow{\text{FC1+ReLU}} (4096) \xrightarrow{\text{FC2+ReLU}} (4096) \xrightarrow{\text{FC3}} (1000)
$$

<span style="font-size: 14px;">**Parameters per layer:**</span>

* <span style="font-size: 14px;">**FC1:** $25{,}088 \times 4{,}096 + 4{,}096 = 102{,}764{,}544$</span>
* <span style="font-size: 14px;">**FC2:** $4{,}096 \times 4{,}096 + 4{,}096 = 16{,}781{,}312$</span>
* <span style="font-size: 14px;">**FC3:** $4{,}096 \times 1{,}000 + 1{,}000 = 4{,}097{,}000$</span>
* <span style="font-size: 14px;">**Total:** $123{,}642{,}856$</span>

---

## <span style="font-size: 16px;">Modern Replacement: Global Average Pooling</span>

<span style="font-size: 14px;">Starting with Network in Network (Lin et al., 2013) and popularized by GoogLeNet (Szegedy et al., 2014), **global average pooling (GAP)** replaced FC classifier heads. GAP averages each feature map across all spatial positions:</span>

$$
z_c = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} F_{i,j,c}
$$

<span style="font-size: 14px;">For VGG16's final volume ($7 \times 7 \times 512$), GAP produces a 512-dimensional vector. A single FC layer maps 512 to num_classes:</span>

* <span style="font-size: 14px;">**VGG16 FC head:** ~123.6 million parameters</span>
* <span style="font-size: 14px;">**GAP + single FC:** $512 \times 1{,}000 + 1{,}000 = 513{,}000$ parameters</span>

<span style="font-size: 14px;">A reduction of approximately **241x**. ResNet (He et al., 2015) adopted this approach, and it became the standard for all subsequent architectures including DenseNet, EfficientNet, and vision transformers. Beyond parameter efficiency, GAP acts as a structural regularizer, makes the network fully convolutional (no fixed input size), and reduces overfitting.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

### <span style="font-size: 14px;">Applying ReLU After FC3</span>

<span style="font-size: 14px;">Applying ReLU after the final layer forces all logits to be non-negative. The network loses the ability to express that a class is unlikely (negative logit = low probability after softmax). Cross-entropy loss expects logits that can be any real number. ReLU after FC3 constrains the output space unnecessarily and leads to worse convergence.</span>

### <span style="font-size: 14px;">Wrong Flatten Dimension</span>

<span style="font-size: 14px;">With batched inputs of shape $(B, C, H, W)$, the flatten must preserve the batch dimension. In PyTorch, use `x.view(x.size(0), -1)` or `torch.flatten(x, start_dim=1)`. Using `x.view(-1)` removes the batch dimension and concatenates all samples into one vector, causing a shape mismatch at the FC layer.</span>

### <span style="font-size: 14px;">Wrong FC1 Input Size</span>

<span style="font-size: 14px;">FC1's input size must exactly match the flattened feature volume. For VGG16 with $224 \times 224$ inputs, this is $7 \times 7 \times 512 = 25{,}088$. A different input size (e.g., $256 \times 256$) changes the feature volume to $8 \times 8 \times 512 = 32{,}768$, causing a runtime dimension mismatch. This rigidity is one reason modern architectures prefer global average pooling.</span>

### <span style="font-size: 14px;">Forgetting Bias</span>

<span style="font-size: 14px;">Each FC layer computes $\mathbf{y} = \mathbf{x}W + \mathbf{b}$. Omitting the bias restricts the layer to transformations through the origin, reducing representational capacity. Frameworks default to including bias (`nn.Linear(in, out, bias=True)`), but from-scratch implementations can miss it.</span>

### <span style="font-size: 14px;">Confusing Logits With Probabilities</span>

<span style="font-size: 14px;">FC3 outputs logits, not probabilities. Logits can be any real number and do not sum to 1. Probabilities require softmax: $p_i = \frac{e^{y_i}}{\sum_{j=1}^{K} e^{y_j}}$. Applying softmax before `nn.CrossEntropyLoss` applies it twice, producing incorrect training behavior. Always track whether you are working with logits or probabilities at each pipeline stage.</span>

---
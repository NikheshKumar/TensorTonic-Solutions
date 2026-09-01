# <span style="font-size: 20px;">Full DenseNet Forward Pass</span>

<span style="font-size: 14px;">DenseNet (Huang et al., 2017) is a convolutional architecture in which every layer receives the feature maps of all preceding layers within a block as input. The full forward pass assembles a stem convolution, several dense blocks separated by transition layers, a final normalization, global average pooling, and a linear classifier into one end to end function from an image to class logits.</span>

---

## <span style="font-size: 16px;">What the Full Network Computes</span>

<span style="font-size: 14px;">A DenseNet maps an input image $x \in \mathbb{R}^{N \times C_{in} \times H \times W}$ to logits $z \in \mathbb{R}^{N \times K}$ over $K$ classes. The paper organizes the network into a small number of dense blocks (typically 3 or 4) joined by transition layers that downsample. The repeating motif is the dense block: a stack of composite layers where layer $\ell$ sees the concatenation of all earlier feature maps.</span>

<span style="font-size: 14px;">The end to end pipeline is:</span>

* <span style="font-size: 14px;">**Stem:** a single $3 \times 3$ convolution that lifts the raw channels to an initial feature width $C_0$ (the paper uses $2k$ for a growth rate $k$ on ImageNet, with a stride and pooling stem there; this problem uses the simpler CIFAR style $3 \times 3$ stem with padding 1).</span>
* <span style="font-size: 14px;">**Dense blocks:** each block grows the channel count by $k$ per composite layer through concatenation.</span>
* <span style="font-size: 14px;">**Transitions:** between consecutive blocks, a batch norm, ReLU, $1 \times 1$ convolution, and $2 \times 2$ average pool reduce both channels and spatial resolution.</span>
* <span style="font-size: 14px;">**Head:** a final batch norm and ReLU, global average pooling over the spatial dimensions, and a linear layer to class logits.</span>

---

## <span style="font-size: 16px;">The Composite Layer</span>

<span style="font-size: 14px;">Each layer inside a block is a **composite function** $H_\ell$. In the plain (non bottleneck) form used here, $H_\ell$ is batch normalization, then ReLU, then a $3 \times 3$ convolution with padding 1 that outputs exactly $k$ feature maps, where $k$ is the **growth rate**:</span>

$$
H_\ell(x) = \text{Conv}_{3\times3}\big(\text{ReLU}(\text{BN}(x))\big)
$$

<span style="font-size: 14px;">The pre activation order (BN then ReLU then convolution) follows the identity mappings work of He et al. (2016). It matters: putting normalization before the convolution keeps the concatenated inputs on a comparable scale even though they originate from layers at very different depths.</span>

<span style="font-size: 14px;">Batch normalization in inference uses the stored running statistics, so for a channel $c$ the normalized activation is:</span>

$$
\hat{x}_c = \frac{x_c - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}}, \qquad y_c = \gamma_c \hat{x}_c + \beta_c
$$

<span style="font-size: 14px;">where $\mu_c, \sigma_c^2$ are the running mean and variance, and $\gamma_c, \beta_c$ are the learned scale and shift. The padding of 1 on the $3 \times 3$ convolution keeps the spatial size unchanged so that all feature maps inside a block remain concatenable.</span>

---

## <span style="font-size: 16px;">Dense Connectivity Inside a Block</span>

<span style="font-size: 14px;">The defining idea of DenseNet is **dense connectivity**. Layer $\ell$ receives the feature maps of all preceding layers as input, formed by concatenation along the channel axis:</span>

$$
x_\ell = H_\ell\big([x_0, x_1, \ldots, x_{\ell-1}]\big)
$$

<span style="font-size: 14px;">Here $x_0$ is the block input and $[\cdot]$ is channel concatenation. A block with $L$ composite layers and growth rate $k$ that starts with $C_0$ channels ends with $C_0 + L \cdot k$ channels. Because each layer adds only $k$ maps, the network stays narrow even though connectivity is dense.</span>

<span style="font-size: 14px;">The implementation maintains a running list of feature maps, appends each layer's $k$ new maps, and concatenates the accumulated list to feed the next layer:</span>

* <span style="font-size: 14px;">Start with the block input as the first element of the feature list.</span>
* <span style="font-size: 14px;">For each composite layer, compute $k$ new maps from the concatenation of everything seen so far.</span>
* <span style="font-size: 14px;">Append the new maps and concatenate again for the next layer.</span>
* <span style="font-size: 14px;">The block output is the concatenation of the input plus every layer's output.</span>

<span style="font-size: 14px;">This is the contrast with ResNet, where the shortcut is additive: $x_\ell = H_\ell(x_{\ell-1}) + x_{\ell-1}$. Addition combines features by summation, which can impede information flow; concatenation preserves every feature map intact and lets later layers selectively reuse them.</span>

---

## <span style="font-size: 16px;">Transition Layers</span>

<span style="font-size: 14px;">Dense blocks keep spatial resolution fixed, so the network needs explicit downsampling between blocks. A **transition layer** does this and also compresses the channel count:</span>

$$
\text{Transition}(x) = \text{AvgPool}_{2\times2}\Big(\text{Conv}_{1\times1}\big(\text{ReLU}(\text{BN}(x))\big)\Big)
$$

<span style="font-size: 14px;">The $1 \times 1$ convolution has no bias and outputs a reduced number of channels. The paper introduces a **compression factor** $\theta \in (0, 1]$: a transition that follows a block with $m$ channels produces $\lfloor \theta m \rfloor$ output channels. DenseNet-BC uses $\theta = 0.5$. The $2 \times 2$ average pool with stride 2 then halves both height and width. Transitions appear between blocks only, never after the last block, so a network with $B$ blocks has exactly $B - 1$ transitions.</span>

---

## <span style="font-size: 16px;">Global Average Pooling and the Classifier</span>

<span style="font-size: 14px;">After the final dense block, the network applies one more batch norm and ReLU, then collapses the spatial dimensions with **global average pooling**. For a feature tensor of shape $(N, C, H, W)$ the pooled vector is:</span>

$$
p_{n,c} = \frac{1}{H W} \sum_{i=1}^{H} \sum_{j=1}^{W} x_{n,c,i,j}
$$

<span style="font-size: 14px;">producing $(N, C)$. Global average pooling, popularized by Network in Network (Lin et al., 2014), replaces large fully connected layers with a single spatial average per channel. It removes a huge number of parameters and imposes a useful structural prior: each channel of the final feature map acts as a confidence map for a concept, and its spatial average is the evidence for that concept.</span>

<span style="font-size: 14px;">The classifier is a single linear layer applied to the pooled vector:</span>

$$
z = p\, W_{fc}^\top + b_{fc}
$$

<span style="font-size: 14px;">with $W_{fc} \in \mathbb{R}^{K \times C}$ and $b_{fc} \in \mathbb{R}^{K}$, giving logits of shape $(N, K)$.</span>

---

## <span style="font-size: 16px;">Standard DenseNet Configurations</span>

<span style="font-size: 14px;">The paper defines a family of networks (Table 1) that all share four dense blocks with growth rate $k = 32$ and the BC variant ($1 \times 1$ bottleneck plus $\theta = 0.5$ compression). They differ only in the number of composite layers per block:</span>

* <span style="font-size: 14px;">**DenseNet-121:** blocks of 6, 12, 24, 16 layers.</span>
* <span style="font-size: 14px;">**DenseNet-169:** blocks of 6, 12, 32, 32 layers.</span>
* <span style="font-size: 14px;">**DenseNet-201:** blocks of 6, 12, 48, 32 layers.</span>
* <span style="font-size: 14px;">**DenseNet-161:** a wider variant with $k = 48$ and blocks of 6, 12, 36, 24 layers.</span>

<span style="font-size: 14px;">The number in the name counts layers with learnable weights: convolutions in the composite layers and transitions, plus the stem and the classifier. The depth grows but the parameter count stays modest because the per layer growth $k$ is small and transitions repeatedly compress the width.</span>

---

## <span style="font-size: 16px;">Bottleneck and Compression in Full Models</span>

<span style="font-size: 14px;">In the full DenseNet-BC, the composite layer is augmented with a bottleneck: a $1 \times 1$ convolution produces $4k$ feature maps before the $3 \times 3$ convolution. The bottleneck caps the cost of the $3 \times 3$ convolution, whose input width grows with every layer. Combined with transition compression $\theta = 0.5$, BC models reach the best accuracy per parameter. This problem deliberately uses the **plain** composite layer (BN-ReLU-$3 \times 3$Conv only) so the forward logic stays tractable; the dense connectivity, transitions, pooling, and classifier are identical to the full model.</span>

---

## <span style="font-size: 16px;">Why Dense Connectivity Helps at Network Scale</span>

<span style="font-size: 14px;">The benefits of concatenative connectivity are clearest when reasoning about the whole network rather than a single block.</span>

* <span style="font-size: 14px;">**Implicit deep supervision.** Because the classifier sees feature maps from many depths through concatenation and short transition paths, gradients reach early layers over short routes. The paper notes this acts like the deep supervision of DSN (Lee et al., 2015) without auxiliary classifiers, easing the training of very deep models.</span>
* <span style="font-size: 14px;">**Feature reuse.** A layer can produce only $k$ new maps yet still draw on the full collection of earlier features. The network does not need to relearn redundant representations at each depth, which is why a 32 channel growth rate is enough for state of the art accuracy.</span>
* <span style="font-size: 14px;">**Diversified gradient flow.** Each layer receives gradients from the loss through every later layer it feeds. This many to many connectivity reduces the chance that a single bad path stalls learning, a structural advantage that compounds as the network deepens.</span>
* <span style="font-size: 14px;">**Regularization on small data.** The paper reports that dense connectivity reduces overfitting on datasets like CIFAR without heavy augmentation, attributing it to the compact, reused feature representation.</span>

<span style="font-size: 14px;">A subtle cost is memory. Naive concatenation stores every intermediate feature map for the backward pass, so memory grows quadratically with block depth. The paper and follow up work address this with shared memory allocations and recomputation, but the forward computation itself is exactly the concatenation described here.</span>

---

## <span style="font-size: 16px;">Implementation Order and Numerical Notes</span>

<span style="font-size: 14px;">The forward pass is sensitive to the order of operations, and several conventions must be matched exactly to reproduce reference logits:</span>

* <span style="font-size: 14px;">**Pre activation within every BN-ReLU-Conv unit.** Both composite layers and transitions normalize first, activate, then convolve. The final head also applies BN then ReLU before pooling. Reordering these (for example convolving before normalizing) changes the output.</span>
* <span style="font-size: 14px;">**Padding keeps blocks concatenable.** The $3 \times 3$ convolutions use padding 1 so that height and width are preserved inside a block. Without it, each layer would shrink the spatial size and concatenation would fail.</span>
* <span style="font-size: 14px;">**Convolutions in the head have no bias.** The transition $1 \times 1$ convolution and the composite $3 \times 3$ convolution carry no bias term; the only bias in the network is in the final linear classifier.</span>
* <span style="font-size: 14px;">**Inference batch norm uses running statistics.** At forward time the normalization divides by $\sqrt{\sigma_c^2 + \epsilon}$ using the stored variance, not batch statistics. A small $\epsilon$ (typically $10^{-5}$) guards against division by near zero variance.</span>
* <span style="font-size: 14px;">**Even spatial sizes before pooling.** Each $2 \times 2$ average pool with stride 2 requires an even input size to halve cleanly. Architectures choose input resolutions so that every pre transition feature map has even height and width.</span>

---

## <span style="font-size: 16px;">Comparison with ResNet Forward</span>

<span style="font-size: 14px;">Both DenseNet and ResNet build very deep networks by giving gradients short paths back to early layers, but the mechanism differs:</span>

* <span style="font-size: 14px;">**ResNet** uses additive skip connections. Each block adds its transformed output to its input. Layer count and feature width are decoupled, but features from different depths are summed, mixing them irreversibly.</span>
* <span style="font-size: 14px;">**DenseNet** uses concatenative connectivity. Every feature map is preserved and made available to all later layers, which encourages **feature reuse** and lets the classifier draw on low and high level features directly.</span>
* <span style="font-size: 14px;">**Parameter efficiency:** because layers are narrow ($k$ around 12 to 48) and features are reused rather than relearned, DenseNet matches ResNet accuracy on ImageNet with substantially fewer parameters and FLOPs.</span>

---

## <span style="font-size: 16px;">Worked Example (tiny network)</span>

<span style="font-size: 14px;">Take $N = 1$, $C_{in} = 2$, $H = W = 8$, stem width $C_0 = 4$, growth rate $k = 2$, two blocks of two layers each, and one transition, with $K = 3$ classes.</span>

* <span style="font-size: 14px;">**Stem:** $3 \times 3$ conv with padding 1 gives a $(1, 4, 8, 8)$ feature map.</span>
* <span style="font-size: 14px;">**Block 1:** layer 1 sees 4 channels and adds 2, giving 6; layer 2 sees 6 and adds 2, giving 8. Output is $(1, 8, 8, 8)$.</span>
* <span style="font-size: 14px;">**Transition:** $1 \times 1$ conv compresses 8 to 4 channels, then $2 \times 2$ average pool halves the spatial size to $(1, 4, 4, 4)$.</span>
* <span style="font-size: 14px;">**Block 2:** starts at 4 channels, adds 2 then 2, ending at 8 channels and shape $(1, 8, 4, 4)$.</span>
* <span style="font-size: 14px;">**Head:** final BN and ReLU, then global average pool over the $4 \times 4$ grid gives $(1, 8)$, and the linear layer maps to $(1, 3)$ logits.</span>

<span style="font-size: 14px;">The key invariant to track at every step is the channel count: it grows by $k$ per composite layer and is reset by each transition.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Applying a transition after the last block.** Transitions sit between blocks. A network with $B$ blocks has $B - 1$ transitions. An off by one that downsamples after the final block changes the channel count fed to the final batch norm and the classifier, producing a shape mismatch or silently wrong logits.</span>
* <span style="font-size: 14px;">**Replacing concatenation with addition or sequential chaining.** Dropping the concatenation collapses dense connectivity into a plain feedforward stack. Shapes may still line up if the growth rate matches the input width, so the bug runs but yields the wrong features and breaks the channel growth the rest of the network expects.</span>
* <span style="font-size: 14px;">**Forgetting the final BN-ReLU before pooling.** The last dense block output is not normalized inside the block. Skipping the final batch norm and ReLU feeds unnormalized, possibly negative activations into global average pooling and shifts every logit.</span>
* <span style="font-size: 14px;">**Flattening instead of global average pooling, or summing instead of averaging.** The classifier expects a per channel average, a vector of length $C$. Flattening spatial dimensions changes the input width to the linear layer, and summing rather than averaging scales features by $H W$, both of which corrupt the logits.</span>
* <span style="font-size: 14px;">**Transposing the classifier weight.** With $W_{fc}$ stored as $(K, C)$, the logits are $p\, W_{fc}^\top + b_{fc}$. Using $W_{fc}$ without the transpose either mismatches dimensions or computes the wrong projection.</span>

---

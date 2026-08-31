# <span style="font-size: 20px;">Composite Layer (BN-ReLU-Conv)</span>

<span style="font-size: 14px;">The composite layer is the elementary computation unit inside a DenseNet (Huang et al., 2017). It is the function $H_\ell(\cdot)$ that every layer in a dense block applies to its input: a Batch Normalization, then a ReLU nonlinearity, then a $3 \times 3$ convolution that produces a small fixed number of new feature maps called the **growth rate**.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">In a Densely Connected Convolutional Network, layer $\ell$ receives the concatenation of the feature maps of all preceding layers as its input. It then computes a nonlinear transformation $H_\ell$ and outputs $k$ new feature maps, where $k$ is the growth rate. The paper defines $H_\ell$ as a **composite function** of three consecutive operations.</span>

<span style="font-size: 14px;">Concretely, $H_\ell$ is the ordered composition:</span>

* <span style="font-size: 14px;">**Batch Normalization (BN):** normalize each channel using its running statistics, then apply the learned affine scale and shift.</span>
* <span style="font-size: 14px;">**ReLU:** apply the rectified linear activation element wise.</span>
* <span style="font-size: 14px;">**Convolution:** a $3 \times 3$ convolution with same padding (padding $= 1$, stride $= 1$, no bias) that maps the input channels down to exactly $k$ output channels.</span>

<span style="font-size: 14px;">This problem implements the basic (non bottleneck) composite layer exactly as it appears in Section 3 of the paper.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">For an input tensor $x \in \mathbb{R}^{N \times C \times H \times W}$, evaluation mode Batch Normalization standardizes each channel $c$ using the stored running mean $\mu_c$ and running variance $\sigma^2_c$, then rescales with learned parameters $\gamma_c$ and $\beta_c$:</span>

$$
\hat{x}_{n,c,i,j} = \frac{x_{n,c,i,j} - \mu_c}{\sqrt{\sigma^2_c + \epsilon}}, \qquad y_{n,c,i,j} = \gamma_c \, \hat{x}_{n,c,i,j} + \beta_c
$$

<span style="font-size: 14px;">where $\epsilon$ is a small constant (default $10^{-5}$) added inside the square root for numerical stability. The activation is then rectified:</span>

$$
z_{n,c,i,j} = \max(0, \, y_{n,c,i,j})
$$

<span style="font-size: 14px;">Finally a $3 \times 3$ convolution with weight $W \in \mathbb{R}^{k \times C \times 3 \times 3}$ and no bias produces $k$ output channels with the spatial size preserved by padding $= 1$:</span>

$$
H_\ell(x)_{n,o,i,j} = \sum_{c=1}^{C} \sum_{a=0}^{2} \sum_{b=0}^{2} W_{o,c,a,b} \; z_{n,c,\, i+a-1,\, j+b-1}
$$

<span style="font-size: 14px;">The output shape is $(N, k, H, W)$. The channel count collapses from $C$ to the growth rate $k$, while the spatial resolution $H \times W$ is unchanged.</span>

---

## <span style="font-size: 16px;">The Pre-Activation Order and Why It Matters</span>

<span style="font-size: 14px;">The ordering BN $\to$ ReLU $\to$ Conv is called **pre-activation**, because the normalization and activation come **before** the convolution rather than after it. DenseNet adopts this order directly from the pre-activation ResNet study (He et al., 2016), which showed that placing BN and ReLU ahead of the weight layer improves gradient flow and gives cleaner identity paths in very deep networks.</span>

<span style="font-size: 14px;">Several properties make pre-activation a natural fit for DenseNet:</span>

* <span style="font-size: 14px;">**Clean concatenation:** because each layer outputs the raw convolution result (not a post-activated one), the concatenated input to later layers is well behaved, and each consumer applies its own BN to renormalize the specific channels it reads.</span>
* <span style="font-size: 14px;">**Per source normalization:** a layer deep in a dense block sees a concatenation of feature maps produced at very different depths and with very different scales. Doing BN first lets that layer renormalize every incoming channel before mixing them, which is essential when inputs come from many sources.</span>
* <span style="font-size: 14px;">**Unimpeded gradient path:** with activation functions sitting before the weights, the shortcut connections formed by concatenation carry gradients backward without being squashed by a trailing ReLU.</span>

<span style="font-size: 14px;">The original (post-activation) ResNet used Conv $\to$ BN $\to$ ReLU. Swapping to BN $\to$ ReLU $\to$ Conv is not cosmetic: it changes which tensor is normalized and which tensor is the layer output, and it produces numerically different results. Getting this order wrong is the single most common implementation error for this problem.</span>

---

## <span style="font-size: 16px;">Growth Rate</span>

<span style="font-size: 14px;">The growth rate $k$ is the number of feature maps each composite layer contributes. It is the output channel count of the $3 \times 3$ convolution. DenseNet uses a surprisingly small growth rate (the paper reports strong ImageNet results with $k = 32$, and CIFAR experiments with $k = 12$), which is one reason the architecture is so parameter efficient.</span>

<span style="font-size: 14px;">Because layer $\ell$ receives the concatenation of all earlier outputs, its input channel count is $C_0 + (\ell - 1) k$, where $C_0$ is the number of channels entering the block. The input width therefore grows linearly with depth, but each layer only adds $k$ new maps. The small per layer contribution is what keeps the total model compact despite the dense connectivity.</span>

<span style="font-size: 14px;">In this problem, the input channel count $C$ is whatever the concatenated input happens to be, and the convolution weight has shape $(k, C, 3, 3)$, so the growth rate is read directly from the first dimension of the weight tensor.</span>

---

## <span style="font-size: 16px;">Same Padding and the 3x3 Convolution</span>

<span style="font-size: 14px;">A $3 \times 3$ kernel with stride $1$ and padding $1$ is a **same** convolution: the output spatial dimensions equal the input spatial dimensions. The general formula for one spatial axis is:</span>

$$
H_{\text{out}} = \left\lfloor \frac{H_{\text{in}} + 2p - k_{\text{size}}}{s} \right\rfloor + 1
$$

<span style="font-size: 14px;">With $k_{\text{size}} = 3$, $p = 1$, $s = 1$ this gives $H_{\text{out}} = H_{\text{in}} + 2 - 3 + 1 = H_{\text{in}}$. Preserving spatial size is mandatory inside a dense block: every layer's output must match the spatial dimensions of every other layer's output so that they can be **concatenated** along the channel axis. If the convolution dropped the padding, the output would shrink by two pixels in each spatial dimension and concatenation would fail.</span>

<span style="font-size: 14px;">The convolution carries **no bias** in this composite layer. The affine shift $\beta$ from the preceding BN already provides a per channel additive degree of freedom, so a separate convolution bias would be redundant. Adding a spurious bias term changes the output and is a common bug.</span>

---

## <span style="font-size: 16px;">Evaluation Mode Batch Normalization</span>

<span style="font-size: 14px;">During training, BN computes the mean and variance from the current mini batch and updates a running estimate. At inference time the layer is frozen: it uses the stored running statistics instead of batch statistics, so the transformation becomes a fixed, deterministic affine map per channel. This problem uses the **eval mode** behavior, which is why the running mean $\mu_c$ and running variance $\sigma^2_c$ are supplied as inputs rather than computed from $x$.</span>

<span style="font-size: 14px;">Eval mode is the correct choice for a from scratch composite layer test, because it removes the dependence on batch size and makes the output reproducible. Note the normalization is applied **per channel**: $\mu$, $\sigma^2$, $\gamma$, and $\beta$ are all vectors of length $C$, and each is broadcast across the batch and spatial dimensions.</span>

<span style="font-size: 14px;">A useful sanity check is the **identity BN** setting: with $\gamma = 1$, $\beta = 0$, $\mu = 0$, $\sigma^2 = 1$, the normalization reduces to $\hat{x} = x / \sqrt{1 + \epsilon}$, which is approximately $x$ for small $\epsilon$. In this case the composite layer is essentially ReLU followed by the convolution.</span>

---

## <span style="font-size: 16px;">Paper Context and Design Decisions</span>

<span style="font-size: 14px;">The paper motivates the composite layer with a single goal: maximize information and gradient flow between layers. Where ResNet adds the outputs of layers through summation, DenseNet **concatenates** them. Summation can impede the flow of information because the identity and the residual are combined into one tensor, whereas concatenation keeps every feature map distinct and accessible. The composite function $H_\ell$ is the operator that turns the accumulated, concatenated state into a fresh set of features.</span>

<span style="font-size: 14px;">Three explicit design decisions in Section 3 shape this layer:</span>

* <span style="font-size: 14px;">**Pre-activation composite function.** The authors define $H_\ell$ as the composition of BN, ReLU, and a $3 \times 3$ convolution, citing the pre-activation ResNet (He et al., 2016) as the source of the ordering. This is a deliberate departure from the original post-activation convolution block.</span>
* <span style="font-size: 14px;">**Small growth rate.** Because every layer adds only $k$ feature maps and all of them feed forward, a small $k$ already provides enough new information per layer. The paper argues that each layer adds its own small set of feature maps to the network's collective knowledge, which the authors describe as a global state that every layer can read.</span>
* <span style="font-size: 14px;">**No dropout, BN as regularizer.** The paper reports that with the strong implicit regularization from dense connectivity and BN, dropout is unnecessary for the datasets studied. The BN inside the composite layer is therefore doing double duty: stabilizing activations and providing regularization.</span>

<span style="font-size: 14px;">These choices together explain why the composite layer looks the way it does: a normalization that can cope with heterogeneous concatenated inputs, an activation, and a small same padded convolution that emits exactly $k$ new maps.</span>

---

## <span style="font-size: 16px;">Pre-Activation vs Post-Activation</span>

<span style="font-size: 14px;">It is worth contrasting the two orderings directly, since they are easy to confuse:</span>

* <span style="font-size: 14px;">**Post-activation (original ResNet, He et al. 2015):** Conv $\to$ BN $\to$ ReLU. The convolution runs first on the raw input, BN normalizes the convolution output, and ReLU is the final operation. The block output is a post-activated tensor.</span>
* <span style="font-size: 14px;">**Pre-activation (DenseNet, He et al. 2016):** BN $\to$ ReLU $\to$ Conv. Normalization and activation run on the input, and the convolution is the final operation. The block output is the raw convolution result.</span>

<span style="font-size: 14px;">The difference matters most in deep networks with many shortcut connections. In the pre-activation form, the path that a feature map takes to a downstream layer is not interrupted by a normalization or activation that depends on the addition or concatenation, so gradients propagate more cleanly. For DenseNet, where each feature map may be read by many later layers, keeping the output un-activated also means each consumer can normalize the channels in the way that suits its own computation.</span>

<span style="font-size: 14px;">A practical consequence: in the pre-activation form the very first layer of a block normalizes its raw input, and the very last operation of every layer is a linear convolution. If an implementation accidentally appends a trailing ReLU after the convolution, the outputs become non negative and concatenation feeds only rectified features forward, which is not what the paper specifies.</span>

---

## <span style="font-size: 16px;">Role Inside a Dense Block</span>

<span style="font-size: 14px;">A dense block stacks several composite layers. Layer $\ell$ takes the concatenation $[x_0, x_1, \ldots, x_{\ell-1}]$ of all previous feature maps, applies $H_\ell$, and produces $x_\ell$, a tensor of exactly $k$ channels. That output is then concatenated onto the running collection for the next layer to consume.</span>

<span style="font-size: 14px;">This dense connectivity is the defining idea of the paper: every layer has direct access to the feature maps of all layers before it, and its own output is passed to all layers after it. The composite layer is the per step computation that makes this possible. It must therefore be self contained, normalize its (possibly heterogeneous) input, and emit a small, spatially aligned set of new features.</span>

<span style="font-size: 14px;">The deeper bottleneck variant (DenseNet-B) inserts a $1 \times 1$ convolution before the $3 \times 3$ convolution to reduce input channels first. This problem implements the basic composite layer without the bottleneck, matching the simplest $H_\ell$ definition in the paper.</span>

---

## Worked Example ($N=1$, $C=2$, $H=W=2$, $k=1$)

<span style="font-size: 14px;">Take a single channel pair with $x[0,0] = \begin{pmatrix} 1 & -1 \ 0 & 2 \end{pmatrix}$ and $x[0,1] = \begin{pmatrix} 0 & 1 \ -2 & 1 \end{pmatrix}$. Let $\gamma = [1, 1]$, $\beta = [0, 0]$, $\mu = [0, 0]$, $\sigma^2 = [1, 1]$, and $\epsilon = 0$.</span>

<span style="font-size: 14px;">1. **BN (identity here):** with these statistics $\hat{x} = x$ and $y = x$, so the tensors pass through unchanged.</span>

<span style="font-size: 14px;">2. **ReLU:** clamp negatives to zero. Channel 0 becomes $\begin{pmatrix} 1 & 0 \ 0 & 2 \end{pmatrix}$ and channel 1 becomes $\begin{pmatrix} 0 & 1 \ 0 & 1 \end{pmatrix}$.</span>

<span style="font-size: 14px;">3. **Convolution:** a $3 \times 3$ same padded kernel slides over the zero padded $2 \times 2$ activations, summing the element wise products across both input channels at every position. The result is a single $2 \times 2$ output map (because $k = 1$), with spatial size preserved by the padding.</span>

<span style="font-size: 14px;">The essential observations: the negatives are removed before the convolution sees them, and the two input channels are fused into one output channel by summing over the channel dimension inside the convolution.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Wrong operation order.** Applying Conv $\to$ ReLU $\to$ BN, or Conv $\to$ BN $\to$ ReLU (the post-activation ResNet order), produces a different result. DenseNet uses pre-activation BN $\to$ ReLU $\to$ Conv. The convolution must be the last step and must consume the rectified, normalized tensor.</span>
* <span style="font-size: 14px;">**Dropping the padding.** A $3 \times 3$ convolution with padding $= 0$ shrinks the spatial size from $H \times W$ to $(H-2) \times (W-2)$. Inside a dense block this breaks concatenation, because outputs of different layers no longer share spatial dimensions. Always use padding $= 1$ for the $3 \times 3$ kernel.</span>
* <span style="font-size: 14px;">**Forgetting epsilon.** Omitting $\epsilon$ inside $\sqrt{\sigma^2 + \epsilon}$ changes the normalization and risks division by zero or instability when a channel variance is tiny. The default $\epsilon = 10^{-5}$ must be added before the square root, not after.</span>
* <span style="font-size: 14px;">**Adding a convolution bias.** The composite layer convolution has no bias; the BN affine shift $\beta$ already supplies the additive term. Introducing a bias vector shifts every output channel and silently produces wrong values that still have the correct shape.</span>

---

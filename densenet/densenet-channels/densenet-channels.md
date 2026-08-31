# <span style="font-size: 20px;">Channel Growth and Compression</span>

<span style="font-size: 14px;">DenseNet (Huang et al., 2017) connects every layer to every other layer in a feed-forward fashion. Inside a dense block each layer adds a fixed number of feature maps called the **growth rate** $k$, so the channel count climbs linearly with depth. **Transition layers** placed between blocks use a **compression factor** $\theta$ to shrink that count back down. This problem tracks the channel count through the whole network using nothing but integer arithmetic.</span>

---

## <span style="font-size: 16px;">Dense Connectivity in One Idea</span>

<span style="font-size: 14px;">In a standard convolutional network the $\ell$-th layer receives only the output of layer $\ell-1$. DenseNet changes this: layer $\ell$ receives the concatenation of the feature maps of all preceding layers in the same block.</span>

<span style="font-size: 14px;">If $x_0$ is the block input and $H_\ell$ is the transformation at layer $\ell$ (batch norm, ReLU, then convolution), the output of layer $\ell$ is:</span>

$$
x_\ell = H_\ell([x_0, x_1, \ldots, x_{\ell-1}])
$$

<span style="font-size: 14px;">The square brackets denote **concatenation along the channel axis**, not addition. This is the key difference from ResNet, where the shortcut adds tensors element-wise. Because DenseNet concatenates, feature maps from early layers stay available to every later layer without being overwritten, which the authors argue improves gradient flow and feature reuse.</span>

<span style="font-size: 14px;">A block of $n$ layers therefore has $n(n+1)/2$ direct connections rather than the $n$ connections of a chain. The paper highlights three benefits from this dense wiring: each layer receives the loss gradient directly through the short paths, alleviating the vanishing-gradient problem in deep networks; features are reused rather than recomputed, reducing redundancy; and the implicit deep supervision lets shallow features influence the final prediction. All of these follow from concatenation keeping the early feature maps intact, which is also exactly why the channel count only ever grows inside a block.</span>

---

## <span style="font-size: 16px;">Growth Rate $k$</span>

<span style="font-size: 14px;">Each $H_\ell$ produces exactly $k$ new feature maps, where $k$ is the **growth rate**. The paper sets $k$ small (for example $k = 12$ or $k = 32$). After $n$ layers inside a dense block, the channel count is:</span>

$$
C_{\text{out}} = C_{\text{in}} + n \cdot k
$$

<span style="font-size: 14px;">where $C_{\text{in}}$ is the number of channels entering the block. The growth is purely additive and linear in the number of layers.</span>

<span style="font-size: 14px;">**Why small $k$ works.** In a conventional network each layer must relearn or carry forward useful features, which forces wide layers. In DenseNet every layer can read the collective knowledge of the block directly through concatenation, so each layer only needs to contribute a thin slice of new information. The paper frames the network state as a global memory that each layer adds $k$ maps to. A growth rate as small as 12 is enough to reach state-of-the-art accuracy, which is a large part of DenseNet's parameter efficiency.</span>

<span style="font-size: 14px;">The growth rate is the single most important width hyperparameter in DenseNet. It is constant across all layers and all blocks in a given model, so the channel arithmetic is fully determined by the stem size, $k$, the per-block layer counts, and $\theta$. Larger $k$ gives each layer more capacity to add but inflates the concatenated state quadratically in cost across the block, so the paper deliberately keeps it small and leans on depth and dense connectivity instead of width.</span>

---

## <span style="font-size: 16px;">The Channel Explosion Problem</span>

<span style="font-size: 14px;">Linear growth sounds modest, but stacking several blocks compounds it. A block of 24 layers with $k = 32$ adds $24 \times 32 = 768$ channels. After two or three such blocks the channel count, and therefore the cost of the next $1 \times 1$ and $3 \times 3$ convolutions, becomes large. Concatenation never removes channels, so the count is monotonically non-decreasing within a block.</span>

<span style="font-size: 14px;">Left unchecked, the feature dimension would balloon and dominate both memory and compute. DenseNet controls this with two mechanisms: bottleneck layers inside each block (the **B** in DenseNet-BC), and compression at the transition layers (the **C**). This problem focuses on the second.</span>

<span style="font-size: 14px;">The cost is concrete. A $3 \times 3$ convolution that maps $C_{\text{in}}$ channels to $C_{\text{out}}$ channels costs on the order of $C_{\text{in}} \cdot C_{\text{out}}$ multiply-accumulates per spatial position, so a block whose input has grown to a thousand channels makes every convolution inside the next block expensive. Compression at the transition resets the input width before the next block begins, which is why $\theta = 0.5$ keeps deep DenseNets practical to train. Tracking the exact channel count at each stage is the first step in budgeting that compute, and it is what this problem asks for.</span>

---

## <span style="font-size: 16px;">Compression $\theta$ at Transition Layers</span>

<span style="font-size: 14px;">A **transition layer** sits between two dense blocks. It applies batch norm, a $1 \times 1$ convolution, and $2 \times 2$ average pooling. The $1 \times 1$ convolution is where compression happens: if the block before it produced $m$ feature maps, the transition outputs:</span>

$$
m_{\text{out}} = \lfloor \theta \cdot m \rfloor
$$

<span style="font-size: 14px;">where $\theta \in (0, 1]$ is the **compression factor**. The paper calls a network with $\theta < 1$ a **DenseNet-C** and uses $\theta = 0.5$ in its experiments, which halves the channel count at every transition. When $\theta = 1$ there is no compression and the transition only changes spatial resolution, not channel count.</span>

<span style="font-size: 14px;">The floor function matters. Channel counts are integers, so $\theta \cdot m$ must be rounded down. With $\theta = 0.5$ and an even $m$ the floor is invisible, but with $\theta = 0.75$ or an odd $m$ it changes the answer. For example $\lfloor 0.75 \cdot 66 \rfloor = \lfloor 49.5 \rfloor = 49$, not 50.</span>

---

## <span style="font-size: 16px;">Where Transitions Sit</span>

<span style="font-size: 14px;">Transitions live **between** dense blocks, never after the final block. The last block's output feeds a global average pool and the classifier, so there is nothing to transition into. For a network with $B$ dense blocks there are exactly $B - 1$ transition layers.</span>

<span style="font-size: 14px;">Reading the channel sequence in order, the stages alternate block, transition, block, transition, and end on a block:</span>

* <span style="font-size: 14px;">**after stem:** the initial convolution output, the starting channel count</span>
* <span style="font-size: 14px;">**after block 1:** stem plus $n_1 \cdot k$</span>
* <span style="font-size: 14px;">**after transition 1:** floor of $\theta$ times the block-1 count</span>
* <span style="font-size: 14px;">**after block 2:** transition-1 count plus $n_2 \cdot k$, and so on</span>
* <span style="font-size: 14px;">**after final block:** no transition follows</span>

<span style="font-size: 14px;">An off-by-one here (adding a transition after the last block, or forgetting one between blocks) is the most common bug when reconstructing the channel schedule.</span>

---

## <span style="font-size: 16px;">Bottleneck Layers and the BC Naming</span>

<span style="font-size: 14px;">Compression is one of two width-control tricks in DenseNet. The other is the **bottleneck**. Inside a dense block each $H_\ell$ can be a single $3 \times 3$ convolution, or it can first apply a $1 \times 1$ convolution that reduces the concatenated input to $4k$ channels before the $3 \times 3$ produces the final $k$ maps. The $1 \times 1$ step is the bottleneck.</span>

<span style="font-size: 14px;">The paper names variants by which tricks are present:</span>

* <span style="font-size: 14px;">**DenseNet:** plain, no bottleneck, no compression ($\theta = 1$).</span>
* <span style="font-size: 14px;">**DenseNet-B:** bottleneck layers inside blocks.</span>
* <span style="font-size: 14px;">**DenseNet-C:** compression at transitions ($\theta < 1$).</span>
* <span style="font-size: 14px;">**DenseNet-BC:** both, the configuration used for the ImageNet models such as DenseNet-121, 169, 201, and 161.</span>

<span style="font-size: 14px;">The bottleneck affects the internal width of a layer but not the block's output channel count: a block of $n$ layers still adds $n \cdot k$ channels regardless of whether each layer used a bottleneck. So for tracking the stage-by-stage channel count, only $k$, the layer counts, and $\theta$ matter. The bottleneck is an internal detail that this problem does not need to model.</span>

---

## <span style="font-size: 16px;">Parameter Efficiency vs ResNet</span>

<span style="font-size: 14px;">DenseNet's headline result is accuracy per parameter. The paper reports that a DenseNet-BC with around 0.8M parameters matches a 1.7M-parameter ResNet on CIFAR, and that DenseNet-201 reaches ResNet-101 accuracy on ImageNet with roughly half the parameters.</span>

* <span style="font-size: 14px;">**ResNet adds, DenseNet concatenates.** ResNet's identity shortcut $x_{\ell} = x_{\ell-1} + F(x_{\ell-1})$ keeps the channel count fixed, so layers can be wide and many. DenseNet keeps each layer thin (width $k$) but accumulates channels.</span>
* <span style="font-size: 14px;">**Feature reuse.** Because every layer sees all earlier feature maps, the network does not need to relearn redundant features, which is the mechanism the authors credit for the parameter savings.</span>
* <span style="font-size: 14px;">**Compression caps the cost.** Halving channels at each transition keeps the concatenated state from exploding, so the savings hold even for deep networks.</span>

---

## Worked Example (DenseNet-121 stem and first stages)

<span style="font-size: 14px;">DenseNet-121 uses a stem of 64 channels, growth rate $k = 32$, block layer counts $[6, 12, 24, 16]$, and compression $\theta = 0.5$. Trace the channel count:</span>

<span style="font-size: 14px;">1. **After stem:** $64$.</span>

<span style="font-size: 14px;">2. **After block 1** ($n_1 = 6$): $64 + 6 \times 32 = 64 + 192 = 256$.</span>

<span style="font-size: 14px;">3. **After transition 1:** $\lfloor 0.5 \times 256 \rfloor = 128$.</span>

<span style="font-size: 14px;">4. **After block 2** ($n_2 = 12$): $128 + 12 \times 32 = 128 + 384 = 512$.</span>

<span style="font-size: 14px;">5. **After transition 2:** $\lfloor 0.5 \times 512 \rfloor = 256$.</span>

<span style="font-size: 14px;">6. **After block 3** ($n_3 = 24$): $256 + 24 \times 32 = 256 + 768 = 1024$.</span>

<span style="font-size: 14px;">7. **After transition 3:** $\lfloor 0.5 \times 1024 \rfloor = 512$.</span>

<span style="font-size: 14px;">8. **After block 4** ($n_4 = 16$): $512 + 16 \times 32 = 512 + 512 = 1024$. No transition follows the last block.</span>

<span style="font-size: 14px;">The full sequence is $[64, 256, 128, 512, 256, 1024, 512, 1024]$. Note the final 1024 channels, which is exactly the feature dimension feeding the classifier in DenseNet-121. Reading the sequence shows the intended rhythm: each block roughly doubles or more than doubles the width, and each transition halves it, so the width oscillates upward in a controlled band rather than running away.</span>

---

## Floor in Action ($\theta = 0.75$)

<span style="font-size: 14px;">Take a stem of 33, $k = 11$, blocks $[3, 7]$, and $\theta = 0.75$. After block 1: $33 + 3 \times 11 = 66$. The single transition gives $\lfloor 0.75 \times 66 \rfloor = \lfloor 49.5 \rfloor = 49$. After block 2: $49 + 7 \times 11 = 126$. Sequence: $[33, 66, 49, 126]$. A naive implementation that keeps $49.5$ as a float, or rounds it to 50, would diverge here and at every later stage.</span>

---

## <span style="font-size: 16px;">Standard Configurations and Modern Context</span>

<span style="font-size: 14px;">The ImageNet DenseNet-BC models share a stem of 64 channels (or $2k$ for the wider variant), $\theta = 0.5$, and differ only in their block layer counts:</span>

* <span style="font-size: 14px;">**DenseNet-121:** $k = 32$, blocks $[6, 12, 24, 16]$.</span>
* <span style="font-size: 14px;">**DenseNet-169:** $k = 32$, blocks $[6, 12, 32, 32]$.</span>
* <span style="font-size: 14px;">**DenseNet-201:** $k = 32$, blocks $[6, 12, 48, 32]$.</span>
* <span style="font-size: 14px;">**DenseNet-161:** $k = 48$, blocks $[6, 12, 36, 24]$, with a wider 96-channel stem.</span>

<span style="font-size: 14px;">The number in the name counts the convolutional layers on the main path: roughly the sum of block layers times the layers per dense unit, plus the stem, transitions, and classifier. The channel schedule computed here is the same accounting that frameworks like torchvision use to size each transition's $1 \times 1$ convolution.</span>

<span style="font-size: 14px;">The concatenation idea reappears in later designs. Feature pyramid networks and U-Net style decoders concatenate skip connections, and dense-style connectivity influenced architectures such as DLA and CondenseNet. The linear-growth plus periodic-compression pattern is a clean, deterministic way to reason about how a network's feature width evolves with depth, which is exactly what this problem exercises.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Skipping the floor.** Computing $\theta \cdot m$ without $\lfloor \cdot \rfloor$ leaves a non-integer channel count. With $\theta = 0.5$ and even counts this happens to be exact, so the bug hides in the easy cases and only surfaces with $\theta = 0.75$ or odd counts. Always floor the transition output.</span>
* <span style="font-size: 14px;">**Adding a transition after the last block.** There are $B - 1$ transitions for $B$ blocks. Emitting a compression step after the final block adds a phantom stage and produces a sequence that is one element too long. Guard the transition with a check that the current block is not the last.</span>
* <span style="font-size: 14px;">**Confusing $n \cdot k$ with $n + k$.** The block adds $n$ layers of $k$ maps each, so the increment is the product $n \cdot k$, not the sum. Using $n + k$ collapses to almost the right answer for small $n$ and is easy to miss.</span>
* <span style="font-size: 14px;">**Rounding instead of flooring.** Python's `round` uses banker's rounding and rounds half values to the nearest even integer, which disagrees with $\lfloor \cdot \rfloor$ whenever the fractional part is at least $0.5$. The paper's $1 \times 1$ convolution produces $\lfloor \theta m \rfloor$ output channels, so floor is the correct operation.</span>

---

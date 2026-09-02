# <span style="font-size: 20px;">ReLU Activation Function</span>

<span style="font-size: 14px;">The Rectified Linear Unit (ReLU) is defined as $f(x) = \max(0, x)$. In the 2012 paper "ImageNet Classification with Deep Convolutional Neural Networks," Krizhevsky, Sutskever, and Hinton showed that replacing sigmoid/tanh with ReLU allowed AlexNet to train several times faster while achieving record-breaking performance on ILSVRC-2012.</span>

---

## <span style="font-size: 16px;">What It Is / What It Does</span>

<span style="font-size: 14px;">An **activation function** injects non-linearity into a neural network. Without it, stacked linear layers collapse into a single linear model regardless of depth.</span>

<span style="font-size: 14px;">ReLU does this with extreme simplicity:</span>

* <span style="font-size: 14px;">**Positive inputs** pass through unchanged</span>
* <span style="font-size: 14px;">**Negative inputs** are set to zero</span>
* <span style="font-size: 14px;">**No learned parameters** and no expensive computations (no exponentials, no divisions)</span>
* <span style="font-size: 14px;">**Element-wise** operation with no interaction between elements</span>

<span style="font-size: 14px;">Each neuron operates linearly, but the network creates a **piecewise linear approximation** of complex functions. A deep ReLU network can produce an exponential number of linear regions.</span>

<span style="font-size: 14px;">In AlexNet, ReLU is applied after every convolutional and fully connected layer, independently to all scalar values in each feature map.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

### <span style="font-size: 14px;">ReLU Definition</span>

$$
f(x) = \max(0, x) = \begin{cases} x & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}
$$

### <span style="font-size: 14px;">ReLU Derivative (Gradient)</span>

$$
f'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x < 0 \end{cases}
$$

<span style="font-size: 14px;">$f'(0)$ is technically undefined. In practice, implementations assign $f'(0) = 0$ with no measurable effect on training.</span>

### <span style="font-size: 14px;">Sigmoid Function (for comparison)</span>

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

<span style="font-size: 14px;">Derivative: $\sigma'(x) = \sigma(x)(1 - \sigma(x))$, max $0.25$ at $x = 0$. Gradient always attenuated by at least 4x per layer, compounding through depth.</span>

### <span style="font-size: 14px;">Tanh Function (for comparison)</span>

$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

<span style="font-size: 14px;">Derivative: $\tanh'(x) = 1 - \tanh^2(x)$, max $1.0$ at $x = 0$, but $< 1$ for all $x \neq 0$ and decays rapidly for $|x| > 2$. Both sigmoid and tanh are **saturating nonlinearities**. ReLU is **non-saturating** for positive inputs: gradient is always exactly 1.</span>

---

## <span style="font-size: 16px;">Mechanics / How It Works</span>

### <span style="font-size: 14px;">Element-wise Operation</span>

<span style="font-size: 14px;">Given input tensor $\mathbf{Z}$, output $\mathbf{A}$ has the same shape with $a_{ij} = \max(0, z_{ij})$. No interactions between elements, no parameters, no state.</span>

### <span style="font-size: 14px;">Behavior on Positive Values</span>

<span style="font-size: 14px;">When $z > 0$, ReLU acts as identity: $f(z) = z$, gradient exactly 1. The upstream gradient passes through unmodified during backpropagation. This is the key property: gradient signal travels backward through many layers without attenuation, as long as neurons are active.</span>

### <span style="font-size: 14px;">Behavior on Negative Values</span>

<span style="font-size: 14px;">When $z \leq 0$, ReLU outputs zero and gradient is zero. This creates **dynamic sparsity**: for any given input, only a subset of neurons are active. Krizhevsky et al. noted this sparsity is computationally beneficial, as the network effectively uses a different sparse sub-network for each input.</span>

### <span style="font-size: 14px;">Gradient Flow During Backpropagation</span>

$$
\frac{\partial L}{\partial z} = \frac{\partial L}{\partial a} \cdot f'(z) = \begin{cases} \frac{\partial L}{\partial a} & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}
$$

<span style="font-size: 14px;">ReLU during backpropagation acts as a **binary mask**: gradients pass where neurons were active, blocked where inactive. No attenuation for active neurons. With sigmoid, every layer multiplies by $\leq 0.25$; after 10 layers: $0.25^{10} \approx 9.5 \times 10^{-7}$. With ReLU, active paths have multiplicative factors of exactly 1.</span>

---

## <span style="font-size: 16px;">Paper Context / Design Decisions</span>

### <span style="font-size: 14px;">Why Krizhevsky et al. Chose ReLU</span>

<span style="font-size: 14px;">In Section 3.1 ("ReLU Nonlinearity"), the authors state: "Deep convolutional neural networks with ReLUs train several times faster than their equivalents with tanh units." Their experiment: a four-layer CNN on CIFAR-10 with ReLU reached 25% training error **six times faster** than tanh (**Figure 1**). They clarify: "This is not a regularization effect, because ReLUs do not saturate."</span>

### <span style="font-size: 14px;">Enabling the Depth of AlexNet</span>

<span style="font-size: 14px;">AlexNet has 5 convolutional + 3 fully connected layers (8 learned layers, ~60M parameters). The paper credits ReLU as a key enabler: without fast ReLU training, they could not have trained this architecture on ImageNet (1.2M images, 1,000 classes) in practical time on two GTX 580 GPUs.</span>

<span style="font-size: 14px;">The authors cite prior work by Jarrett et al. (2009) and Nair and Hinton (2010), but AlexNet was the first to validate ReLU at scale on a large, challenging dataset, making it the de facto standard.</span>

### <span style="font-size: 14px;">The Broader Architectural Impact</span>

<span style="font-size: 14px;">ReLU interacted with other AlexNet design choices. Its sparsity complemented dropout (both encourage distributed representations). Its unbounded positive output made local response normalization (LRN) useful for preventing dominant activations among neighboring neurons.</span>

---

## <span style="font-size: 16px;">The Vanishing Gradient Problem</span>

### <span style="font-size: 14px;">Why Sigmoid and Tanh Suffer</span>

<span style="font-size: 14px;">In a deep network with $L$ layers, the gradient of the loss w.r.t. weights in layer $l$ involves a product through all subsequent layers:</span>

$$
\frac{\partial L}{\partial \mathbf{W}^{(l)}} = \frac{\partial L}{\partial \mathbf{a}^{(L)}} \cdot \prod_{k=l+1}^{L} \text{diag}(\sigma'(\mathbf{z}^{(k)})) \cdot \mathbf{W}^{(k)} \cdot \frac{\partial \mathbf{a}^{(l)}}{\partial \mathbf{W}^{(l)}}
$$

<span style="font-size: 14px;">For **sigmoid**, $\sigma'(x) \leq 0.25$. After 8 layers: $0.25^8 \approx 1.5 \times 10^{-5}$, so early-layer gradients are five orders of magnitude smaller than at the output.</span>

<span style="font-size: 14px;">For **tanh**, $\tanh'(0) = 1$ but drops rapidly: $\tanh'(1) \approx 0.42$, $\tanh'(2) \approx 0.07$, $\tanh'(3) \approx 0.01$. Unless all pre-activations cluster near zero (unrealistic), gradients still vanish exponentially.</span>

### <span style="font-size: 14px;">How ReLU Solves This</span>

<span style="font-size: 14px;">For active neurons ($z > 0$), $f'(z) = 1$. The product along any path of active neurons:</span>

$$
\prod_{k=l+1}^{L} f'(z^{(k)}) = 1^{L-l} = 1
$$

<span style="font-size: 14px;">Zero attenuation from the activation function -- a qualitative change from sigmoid/tanh. Inactive neurons ($z \leq 0$) have zero gradient (the "dying ReLU" problem, see Pitfalls), but in a healthy network sufficient neurons remain active for gradient flow through alternative paths.</span>

### <span style="font-size: 14px;">The Mathematical Argument</span>

<span style="font-size: 14px;">Under typical conditions, $\mathbb{E}[|\sigma'(z)|] \approx 0.2$ for sigmoid. For ReLU with ~50% active neurons, $\mathbb{E}[f'(z)] \approx 0.5$. Over 8 layers: $0.5^8 / 0.2^8 = 3.9 \times 10^{-3} / 2.56 \times 10^{-6} \approx 1500\times$ more gradient reaching early layers. Moreover, ReLU gradients that pass through arrive at full strength (factor of 1), unlike sigmoid's always-diminished gradients.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

### <span style="font-size: 14px;">Setup</span>

<span style="font-size: 14px;">Pre-activation values from a single layer:</span>

$$
\mathbf{z} = [-2.0, \; -0.5, \; 0.0, \; 0.3, \; 1.5, \; -1.0, \; 2.7, \; 0.1]
$$

### <span style="font-size: 14px;">Step 1: Forward Pass (Computing ReLU Output)</span>

<span style="font-size: 14px;">Apply $f(x) = \max(0, x)$ element-wise:</span>

* <span style="font-size: 14px;">$f(-2.0) = 0.0$, $f(-0.5) = 0.0$, $f(0.0) = 0.0$ (negative/zero, zeroed out)</span>
* <span style="font-size: 14px;">$f(0.3) = 0.3$, $f(1.5) = 1.5$, $f(2.7) = 2.7$, $f(0.1) = 0.1$ (positive, pass through)</span>
* <span style="font-size: 14px;">$f(-1.0) = 0.0$ (negative, zeroed out)</span>

<span style="font-size: 14px;">**ReLU output:**</span>

$$
\mathbf{a} = [0.0, \; 0.0, \; 0.0, \; 0.3, \; 1.5, \; 0.0, \; 2.7, \; 0.1]
$$

<span style="font-size: 14px;">4 out of 8 values (50%) are zeroed out -- typical and desirable sparsity.</span>

### <span style="font-size: 14px;">Step 2: Backward Pass (Computing ReLU Gradient)</span>

<span style="font-size: 14px;">Upstream gradient:</span>

$$
\frac{\partial L}{\partial \mathbf{a}} = [0.4, \; -0.2, \; 0.1, \; -0.6, \; 0.8, \; 0.3, \; -0.5, \; 0.9]
$$

<span style="font-size: 14px;">Apply binary mask (pass gradient where $z > 0$, block where $z \leq 0$):</span>

* <span style="font-size: 14px;">$z_1=-2.0 \leq 0 \rightarrow 0.0$; $z_2=-0.5 \leq 0 \rightarrow 0.0$; $z_3=0.0 \leq 0 \rightarrow 0.0$; $z_6=-1.0 \leq 0 \rightarrow 0.0$</span>
* <span style="font-size: 14px;">$z_4=0.3 > 0 \rightarrow -0.6$; $z_5=1.5 > 0 \rightarrow 0.8$; $z_7=2.7 > 0 \rightarrow -0.5$; $z_8=0.1 > 0 \rightarrow 0.9$</span>

<span style="font-size: 14px;">**Gradient passed to previous layer:**</span>

$$
\frac{\partial L}{\partial \mathbf{z}} = [0.0, \; 0.0, \; 0.0, \; -0.6, \; 0.8, \; 0.0, \; -0.5, \; 0.9]
$$

<span style="font-size: 14px;">Every active neuron passes its gradient at **full magnitude** -- no attenuation.</span>

### <span style="font-size: 14px;">Step 3: Comparison with Sigmoid</span>

<span style="font-size: 14px;">Sigmoid gradient factors $\sigma'(z) = \sigma(z)(1-\sigma(z))$ for the same inputs:</span>

* <span style="font-size: 14px;">$z=-2.0$: $0.105$; $z=-0.5$: $0.235$; $z=0.0$: $0.250$; $z=0.3$: $0.245$</span>
* <span style="font-size: 14px;">$z=1.5$: $0.149$; $z=-1.0$: $0.197$; $z=2.7$: $0.059$; $z=0.1$: $0.249$</span>

<span style="font-size: 14px;">Even the best case ($z=0$) attenuates to 25%. At $z=2.7$, gradient is reduced to 5.9%. These attenuations compound catastrophically over multiple layers.</span>

### <span style="font-size: 14px;">Step 4: Comparison with Tanh</span>

<span style="font-size: 14px;">Tanh gradient factors $1 - \tanh^2(z)$:</span>

* <span style="font-size: 14px;">$z=-2.0$: $0.071$; $z=-0.5$: $0.786$; $z=0.0$: $1.000$; $z=0.3$: $0.915$</span>
* <span style="font-size: 14px;">$z=1.5$: $0.181$; $z=-1.0$: $0.420$; $z=2.7$: $0.020$; $z=0.1$: $0.990$</span>

<span style="font-size: 14px;">Tanh reaches 1.0 at $z=0$ but drops to 0.020 at $z=2.7$. ReLU gives exactly 1.0 for all positive inputs regardless of magnitude: $f'(100) = f'(0.001) = 1$.</span>

---

## <span style="font-size: 16px;">Variants and Modern Context</span>

### <span style="font-size: 14px;">Leaky ReLU</span>

$$
f(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{if } x \leq 0 \end{cases}
$$

<span style="font-size: 14px;">Typically $\alpha = 0.01$. Ensures every neuron always has a non-zero gradient, preventing complete "death." Proposed by Maas et al. (2013), popular in GANs.</span>

### <span style="font-size: 14px;">Parametric ReLU (PReLU)</span>

$$
f(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha_i x & \text{if } x \leq 0 \end{cases}
$$

<span style="font-size: 14px;">He et al. (2015) made $\alpha_i$ a learnable per-channel parameter. Combined with He/Kaiming initialization (which accounts for ReLU's variance properties), PReLU achieved superhuman ImageNet performance. Learned $\alpha$ values typically converge to small positive numbers.</span>

### <span style="font-size: 14px;">Exponential Linear Unit (ELU)</span>

$$
f(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha(e^x - 1) & \text{if } x \leq 0 \end{cases}
$$

<span style="font-size: 14px;">Clevert et al. (2016), typically $\alpha = 1.0$. Produces negative outputs (pushing mean closer to zero), saturates to $-\alpha$ for large negatives. More expensive than ReLU due to exponential computation; practical benefits often marginal.</span>

### <span style="font-size: 14px;">GELU (Gaussian Error Linear Unit)</span>

$$
f(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]
$$

<span style="font-size: 14px;">Hendrycks and Gimpel (2016). A smooth, probabilistic version of ReLU: scales each input by the probability a standard normal variable would be less than that input. Used in BERT, GPT-2, GPT-3, and most modern transformers.</span>

### <span style="font-size: 14px;">Swish / SiLU (Sigmoid Linear Unit)</span>

$$
f(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
$$

<span style="font-size: 14px;">Ramachandran et al. (2017), Google Brain. Smooth, non-monotonic, unbounded above. Consistently outperformed ReLU across architectures. Used in EfficientNet. Closely related to GELU in practice.</span>

### <span style="font-size: 14px;">Why ReLU Remains Dominant</span>

* <span style="font-size: 14px;">**Simplicity:** trivial to implement, no hyperparameters</span>
* <span style="font-size: 14px;">**Efficiency:** single comparison per element vs. exponentials/error functions</span>
* <span style="font-size: 14px;">**Well-understood initialization:** He initialization is specifically designed and validated for ReLU</span>
* <span style="font-size: 14px;">**Robustness:** rarely fails catastrophically, strong performance across tasks and architectures</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

### <span style="font-size: 14px;">The Dying ReLU Problem</span>

<span style="font-size: 14px;">A neuron "dies" when its weights evolve such that $z = \mathbf{w}^T \mathbf{x} + b < 0$ for all training inputs. Output and gradient are permanently zero -- the neuron never recovers. Most likely with high learning rates or negative bias initialization. Mitigation: zero/small-positive bias init, moderate learning rates, or use Leaky ReLU/PReLU/ELU.</span>

### <span style="font-size: 14px;">Non-Zero-Centered Outputs</span>

<span style="font-size: 14px;">ReLU outputs are always $\geq 0$, so all inputs to the next layer are positive. This forces all weight gradients for that neuron to share the same sign, constraining optimization to zig-zag paths in weight space. Batch normalization largely mitigates this by re-centering activations.</span>

### <span style="font-size: 14px;">Non-Differentiability at Zero</span>

<span style="font-size: 14px;">ReLU has a kink at $x = 0$. In practice this is a non-issue: the probability of exactly $0.0$ in floating-point is negligible, and most implementations set $f'(0) = 0$. ReLU networks converge under standard conditions despite this.</span>

### <span style="font-size: 14px;">Unbounded Activations</span>

<span style="font-size: 14px;">Unlike sigmoid (0 to 1) or tanh (-1 to 1), ReLU has no upper bound. Activations can grow large, risking numerical overflow and unstable gradients. Batch normalization, layer normalization, and weight decay are important companions. In AlexNet, LRN served this role.</span>

### <span style="font-size: 14px;">Effect on Batch Statistics</span>

<span style="font-size: 14px;">ReLU zeros ~50% of inputs, shifting mean upward and reducing variance. He initialization compensates by scaling weights by $\sqrt{2/n}$ instead of $\sqrt{1/n}$. If batch norm is applied before ReLU, the normalized distribution is truncated to a half-normal (mean $\approx 0.4$, variance $\approx 0.68$). The ordering of batch norm and ReLU remains a subject of practical consideration.</span>

### <span style="font-size: 14px;">Sensitivity to Weight Initialization</span>

<span style="font-size: 14px;">ReLU behavior depends on the sign of pre-activations, making initialization critical. He initialization (variance $2/n$) keeps activation variance constant across layers. Xavier/Glorot initialization (variance $1/n$), designed for sigmoid/tanh, causes variance to shrink by 2x at each ReLU layer, leading to vanishing activations in deep networks. Wrong initialization can cause complete training failure.</span>

---
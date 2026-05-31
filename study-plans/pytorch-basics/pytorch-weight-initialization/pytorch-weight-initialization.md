<span style="font-size: 14px;">Weight initialization determines the starting point of optimization. When all weights begin at zero, every neuron in a layer computes the same output, gradients are identical, and the network cannot break symmetry - it effectively has one neuron per layer regardless of width. Random initialization solves symmetry breaking, but the scale of the random values matters enormously.</span>

## <span style="font-size: 14px;">The Variance Problem</span>

<span style="font-size: 14px;">Consider a layer</span> $y = Wx$ <span style="font-size: 14px;">where</span> $W$ <span style="font-size: 14px;">has shape</span> $(n_{out}, n_{in})$<span style="font-size: 14px;">. If the input elements have variance</span> $\text{Var}(x)$ <span style="font-size: 14px;">and the weights are independent with variance</span> $\text{Var}(w)$<span style="font-size: 14px;">, then each output element has variance:</span>

$$
\text{Var}(y_j) = n_{in} \cdot \text{Var}(w) \cdot \text{Var}(x)
$$

<span style="font-size: 14px;">In a deep network with</span> $L$ <span style="font-size: 14px;">layers, the variance after</span> $L$ <span style="font-size: 14px;">layers scales as</span> $(n \cdot \text{Var}(w))^L$<span style="font-size: 14px;">. If</span> $n \cdot \text{Var}(w) > 1$<span style="font-size: 14px;">, activations explode exponentially. If</span> $n \cdot \text{Var}(w) < 1$<span style="font-size: 14px;">, they vanish exponentially.</span>

## <span style="font-size: 14px;">Xavier Initialization (Glorot & Bengio, 2010)</span>

<span style="font-size: 14px;">Xavier initialization maintains the variance of activations across layers when using symmetric activations like sigmoid or tanh. The key insight: set</span> $\text{Var}(w) = \frac{2}{n_{in} + n_{out}}$ <span style="font-size: 14px;">which is a compromise between preserving variance in the forward pass (needs</span> $1/n_{in}$<span style="font-size: 14px;">) and the backward pass (needs</span> $1/n_{out}$<span style="font-size: 14px;">).</span>

* <span style="font-size: 14px;">Uniform variant:</span> $W \sim U\left(-\sqrt{\frac{6}{n_{in}+n_{out}}},\; \sqrt{\frac{6}{n_{in}+n_{out}}}\right)$
* <span style="font-size: 14px;">Normal variant:</span> $W \sim \mathcal{N}\left(0,\; \sqrt{\frac{2}{n_{in}+n_{out}}}\right)$

<span style="font-size: 14px;">The factor of 6 in the uniform bound comes from the variance of</span> $U(-a, a)$ <span style="font-size: 14px;">being</span> $a^2/3$<span style="font-size: 14px;">, so</span> $a = \sqrt{3 \cdot \text{Var}} = \sqrt{\frac{6}{n_{in}+n_{out}}}$<span style="font-size: 14px;">.</span>

## <span style="font-size: 14px;">He Initialization (He et al., 2015)</span>

<span style="font-size: 14px;">ReLU activations zero out roughly half of their inputs, effectively halving the variance at each layer. He initialization compensates by using a larger weight variance that only depends on</span> $n_{in}$<span style="font-size: 14px;">:</span>

$$
\text{Var}(w) = \frac{2}{n_{in}}
$$

* <span style="font-size: 14px;">Uniform variant:</span> $W \sim U\left(-\sqrt{\frac{6}{n_{in}}},\; \sqrt{\frac{6}{n_{in}}}\right)$
* <span style="font-size: 14px;">Normal variant:</span> $W \sim \mathcal{N}\left(0,\; \sqrt{\frac{2}{n_{in}}}\right)$

## <span style="font-size: 14px;">When to Use Which</span>

| <span style="font-size: 14px;">Activation</span> | <span style="font-size: 14px;">Recommended Init</span> |
|---|---|
| <span style="font-size: 14px;">Sigmoid, Tanh</span> | <span style="font-size: 14px;">Xavier (Glorot)</span> |
| <span style="font-size: 14px;">ReLU, Leaky ReLU, ELU</span> | <span style="font-size: 14px;">He (Kaiming)</span> |

## <span style="font-size: 14px;">Common Mistakes</span>

* <span style="font-size: 14px;">Initializing all weights to zero: breaks symmetry, the network cannot learn</span>
* <span style="font-size: 14px;">Using too-small standard deviation: gradients vanish in deep networks</span>
* <span style="font-size: 14px;">Using too-large standard deviation: activations saturate, gradients explode</span>
* <span style="font-size: 14px;">Using Xavier init with ReLU: underestimates the variance needed because ReLU halves it</span>
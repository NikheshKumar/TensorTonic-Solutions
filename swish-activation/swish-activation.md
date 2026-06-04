## The Search for Better Activations

ReLU ($\max(0, x)$) is the most popular activation function in deep learning, but it is not optimal in every setting. Its sharp corner at zero and completely dead negative region leave room for improvement.

Researchers have tried many alternatives (Leaky ReLU, ELU, PReLU, etc.), each designed by hand. In 2017, Google Brain took a different approach: they used **automated search** to discover new activation functions by combining basic mathematical operations. The best function they found was Swish.

---

## What Swish Does

Swish is defined as:

$$
\text{Swish}(x) = x \cdot \sigma(x)
$$

where $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the sigmoid function.

The formula multiplies the input $x$ by its own sigmoid. Since sigmoid outputs values between 0 and 1, Swish acts like a **smooth gate** that controls how much of $x$ passes through:

- When $x$ is large and positive: $\sigma(x) \approx 1$, so $\text{Swish}(x) \approx x$ (passes through fully)
- When $x = 0$: $\sigma(0) = 0.5$, so $\text{Swish}(0) = 0$
- When $x$ is large and negative: $\sigma(x) \approx 0$, so $\text{Swish}(x) \approx 0$ (suppressed)
- When $x$ is slightly negative: the output is a **small negative value** (not zero)

Some concrete values:

- $\text{Swish}(5.0) = 5.0 \times 0.993 \approx 4.966$
- $\text{Swish}(1.0) = 1.0 \times 0.731 = 0.731$
- $\text{Swish}(0) = 0$
- $\text{Swish}(-1.0) = -1.0 \times 0.269 = -0.269$
- $\text{Swish}(-5.0) = -5.0 \times 0.007 \approx -0.034$

---

## The Non-Monotonic "Bump"

One of Swish's most interesting properties: it is **non-monotonic**. Unlike ReLU (always increasing or flat) or sigmoid (always increasing), Swish dips slightly below zero before coming back up.

The minimum occurs at approximately $x \approx -1.28$, where $\text{Swish}(x) \approx -0.278$.

This means:

- For $x < -1.28$: the output is actually increasing (becoming less negative) as $x$ decreases further
- The curve has a small "bump" in the negative region

This non-monotonicity is unusual for an activation function, and it is part of what makes Swish work well. It allows the network to produce small negative activations for moderately negative inputs while still suppressing very negative ones. This provides more information to downstream layers compared to ReLU's hard zero.

---

## Swish vs. ReLU

The key differences:

**Smoothness**: Swish is infinitely differentiable (smooth everywhere). ReLU has a sharp corner at zero where the derivative is undefined. Smooth functions create smoother loss landscapes, which can make optimization easier.

**Negative values**: Swish allows small negative outputs. ReLU outputs exactly zero for all negative inputs. The small negative values in Swish:
- Prevent dead neurons (the gradient is never exactly zero for finite inputs)
- Allow the network to represent and propagate negative signals
- Act as a form of implicit regularization

**Asymptotic behavior**: for large positive $x$, both approach $f(x) \approx x$. For large negative $x$, both approach 0. The difference is in the transition region around zero.

**Computational cost**: Swish requires computing sigmoid ($e^{-x}$, addition, division) plus a multiplication. ReLU is just a comparison with zero. Swish is more expensive per operation, but the overall impact on training time is small because the matrix multiplications dominate.

---

## The Gradient

The derivative of Swish is:

$$
\text{Swish}'(x) = \sigma(x) + x \cdot \sigma(x)(1 - \sigma(x))
$$

This can be simplified to:

$$
\text{Swish}'(x) = \sigma(x) + x \cdot \sigma'(x) = \text{Swish}(x) + \sigma(x)(1 - \text{Swish}(x))
$$

Key properties of the gradient:

- At $x = 0$: gradient $= 0.5 + 0 = 0.5$
- For large positive $x$: gradient approaches 1 (like ReLU)
- For large negative $x$: gradient approaches 0 (like ReLU)
- The gradient is **never exactly zero** for finite inputs, so no neuron can completely die

---

## Swish and SiLU

Swish is also known as **SiLU** (Sigmoid Linear Unit). The names refer to the same function:

$$
\text{SiLU}(x) = \text{Swish}(x) = x \cdot \sigma(x)
$$

In PyTorch, it is called \`torch.nn.SiLU\`. In some papers and frameworks, you will see either name. They are interchangeable.

---

## Where Swish Shows Up

- **EfficientNet**: Google's state-of-the-art image classification architecture uses Swish throughout instead of ReLU. This was one of the first major models to adopt Swish.
- **Vision Transformers**: many ViT variants use Swish/SiLU in their MLP layers
- **Diffusion models**: Swish is commonly used in the U-Net architectures of image generation models (Stable Diffusion, DALL-E)
- **Mobile architectures**: MobileNetV3 and other efficient architectures use a related variant called hard-swish ($x \cdot \frac{\text{ReLU6}(x+3)}{6}$) that approximates Swish with cheaper operations
- **General replacement for ReLU**: in many benchmarks, Swish matches or outperforms ReLU, especially in deeper networks. It is a safe default choice when you want something better than ReLU without much tuning.
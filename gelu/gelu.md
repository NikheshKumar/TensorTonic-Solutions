## Why Activation Functions Exist

A neural network without activation functions is just a stack of linear transformations. No matter how many layers you add, the whole thing collapses into a single linear function: $y = Wx + b$. It cannot learn curves, boundaries, or any non-trivial pattern.

Activation functions introduce **nonlinearity** after each layer. They decide which neurons "fire" and how strongly. The choice of activation function has a major impact on how well the network trains and how it behaves.

---

## ReLU: The Current Standard

The most widely used activation function is **ReLU** (Rectified Linear Unit):

$$
\text{ReLU}(x) = \max(0, x)
$$

- Positive inputs pass through unchanged
- Negative inputs get set to exactly zero
- Simple, fast, and works well in most cases

But ReLU has a sharp corner at $x = 0$. The output jumps from 0 to linear with no smooth transition. The gradient is either 0 (for negative inputs) or 1 (for positive inputs), with a discontinuity at zero.

This causes two issues:

- **Dead neurons**: if a neuron's input is always negative, its gradient is always 0, so it never updates. It is permanently "dead."
- **Non-smooth gradients**: the sharp transition at zero can make optimization less stable

---

## GELU: A Smooth Alternative

GELU (Gaussian Error Linear Unit) replaces ReLU's hard cutoff with a **smooth, probabilistic** transition. The formula is:

$$
\text{GELU}(x) = x \cdot \Phi(x)
$$

where $\Phi(x)$ is the **cumulative distribution function (CDF)** of the standard normal distribution. $\Phi(x)$ answers the question: "If I draw a random number from a standard normal distribution $N(0,1)$, what is the probability that it is less than $x$?"

Some values of $\Phi(x)$ to build intuition:

- $\Phi(-3) \approx 0.001$ (almost zero)
- $\Phi(-1) \approx 0.159$
- $\Phi(0) = 0.5$ (exactly half)
- $\Phi(1) \approx 0.841$
- $\Phi(3) \approx 0.999$ (almost one)

So GELU multiplies each input $x$ by the probability that a Gaussian random variable is less than $x$:

- **Large positive** $x$: $\Phi(x) \approx 1$, so $\text{GELU}(x) \approx x$ (passes through almost unchanged)
- **Large negative** $x$: $\Phi(x) \approx 0$, so $\text{GELU}(x) \approx 0$ (suppressed, like ReLU)
- **Near zero**: $\Phi(x) \approx 0.5$, so the output is smoothly scaled. For example, $\text{GELU}(0) = 0 \times 0.5 = 0$

The transition from "suppress" to "pass through" is **gradual**, not abrupt like ReLU.

---

## The Error Function Connection

The standard normal CDF can be written in terms of the **error function** (erf):

$$
\Phi(x) = \frac{1}{2}\left(1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right)
$$

Substituting into the GELU formula:

$$
\text{GELU}(x) = \frac{1}{2} x \left(1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right)
$$

The error function $\text{erf}(z)$ is a standard mathematical function that goes smoothly from $-1$ to $+1$:

- $\text{erf}(-\infty) = -1$
- $\text{erf}(0) = 0$
- $\text{erf}(+\infty) = +1$

It is available in most math libraries and can be computed efficiently.

---

## Comparing GELU to ReLU

Some concrete values to see the difference:

**At $x = -1$:**
- ReLU: $\max(0, -1) = 0$ (hard zero)
- GELU: $-1 \times \Phi(-1) = -1 \times 0.159 = -0.159$ (small negative value)

**At $x = 0$:**
- ReLU: $0$
- GELU: $0 \times 0.5 = 0$

**At $x = 1$:**
- ReLU: $1$
- GELU: $1 \times \Phi(1) = 1 \times 0.841 = 0.841$ (slightly less than 1)

**At $x = 3$:**
- ReLU: $3$
- GELU: $3 \times \Phi(3) = 3 \times 0.999 \approx 2.996$ (almost identical)

Key differences:

- GELU allows **small negative outputs** for negative inputs (the minimum is about $-0.17$ at $x \approx -0.75$). ReLU always outputs exactly 0 for negative inputs.
- GELU is **smooth everywhere**. ReLU has a kink at $x = 0$.
- For large positive inputs, both behave almost identically.
- GELU never has "dead neurons" because the gradient is never exactly zero for any finite input.

---

## Why Smoothness Matters

The smoothness of GELU has practical consequences for training:

- **Gradients flow everywhere**: even for slightly negative inputs, the gradient is nonzero. No neurons die permanently.
- **Better optimization landscape**: the smooth curve means the loss surface has fewer sharp edges, making gradient-based optimization more stable.
- **Information preservation**: slightly negative signals carry information. Hard-zeroing them (like ReLU) throws that information away.

---

## Where GELU Is Used

GELU has become the **standard activation for Transformer models**:

- **BERT**: uses GELU in its feed-forward layers (this is what popularized GELU)
- **GPT-2, GPT-3**: use GELU throughout
- **Vision Transformers (ViT)**: use GELU
- **Most modern Transformers**: default to GELU over ReLU

The original GELU paper (Hendrycks and Gimpel, 2016) showed that GELU consistently outperforms ReLU on several benchmarks, especially in natural language processing tasks.
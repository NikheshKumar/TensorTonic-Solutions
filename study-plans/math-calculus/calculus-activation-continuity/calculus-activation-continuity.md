<span style="font-size: 14px;">Continuity and differentiability are two of the most important properties that distinguish activation functions in neural networks. These concepts from calculus directly influence gradient flow, training stability, and the optimization landscape. Understanding where an activation function has "kinks" (non-differentiable points) explains many practical phenomena in deep learning.</span>

## <span style="font-size: 14px;">Continuity: Definition and Intuition</span>

<span style="font-size: 14px;">A function $f$ is **continuous** at a point $x = a$ if three conditions hold simultaneously:</span>

* <span style="font-size: 14px;">$f(a)$ is defined (the function has a value at $a$)</span>
* <span style="font-size: 14px;">$\lim_{x \to a} f(x)$ exists (the limit from both sides agrees)</span>
* <span style="font-size: 14px;">$\lim_{x \to a} f(x) = f(a)$ (the limit equals the function value)</span>

<span style="font-size: 14px;">Equivalently, using one-sided limits:</span>

$$
\lim_{x \to a^-} f(x) = f(a) = \lim_{x \to a^+} f(x)
$$

<span style="font-size: 14px;">Intuitively, a continuous function has no "jumps" or "holes" in its graph. You can draw it without lifting your pen. For activation functions in neural networks, continuity is essential because it ensures that small changes in input produce small changes in output, which is necessary for stable forward propagation.</span>

### <span style="font-size: 14px;">Types of Discontinuity</span>

<span style="font-size: 14px;">There are several types of discontinuity, though they rarely appear in standard activation functions:</span>

* <span style="font-size: 14px;">**Jump discontinuity**: the left and right limits exist but are not equal. Example: the Heaviside step function $H(x)$ jumps from 0 to 1 at $x = 0$</span>
* <span style="font-size: 14px;">**Removable discontinuity**: the limit exists but the function is either not defined at that point or defined to a different value</span>
* <span style="font-size: 14px;">**Essential discontinuity**: the function oscillates or blows up near the point (e.g., $\sin(1/x)$ near $x = 0$)</span>

<span style="font-size: 14px;">All standard activation functions used in modern deep learning (ReLU, Leaky ReLU, GELU, Sigmoid, Tanh, Swish, etc.) are continuous everywhere. This is by design: a discontinuous activation would create sudden jumps in the network output, making optimization nearly impossible.</span>

## <span style="font-size: 14px;">Differentiability: Definition and Intuition</span>

<span style="font-size: 14px;">A function $f$ is **differentiable** at $x = a$ if the following limit exists:</span>

$$
f'(a) = \lim_{h \to 0} \frac{f(a + h) - f(a)}{h}
$$

<span style="font-size: 14px;">This is equivalent to requiring that the **left derivative** and **right derivative** exist and are equal:</span>

$$
f'_-(a) = \lim_{h \to 0^+} \frac{f(a) - f(a - h)}{h} \quad \text{(left derivative)}
$$

$$
f'_+(a) = \lim_{h \to 0^+} \frac{f(a + h) - f(a)}{h} \quad \text{(right derivative)}
$$

$$
f \text{ is differentiable at } a \iff f'_-(a) = f'_+(a)
$$

<span style="font-size: 14px;">Key relationship: **differentiability implies continuity, but continuity does not imply differentiability**. A function can be continuous everywhere but have "kinks" or "corners" where it is not differentiable. This is exactly the situation with ReLU.</span>

### <span style="font-size: 14px;">Numerical Verification</span>

<span style="font-size: 14px;">In practice, we can numerically approximate derivatives using a small step size $h$:</span>

$$
f'_-(a) \approx \frac{f(a) - f(a - h)}{h}, \quad f'_+(a) \approx \frac{f(a + h) - f(a)}{h}
$$

<span style="font-size: 14px;">If $|f'_-(a) - f'_+(a)| < \epsilon$ for some small tolerance $\epsilon$, we consider the function differentiable at $a$. Both $h$ and $\epsilon$ should be chosen small enough to approximate the true derivative while avoiding floating-point cancellation.</span>

## <span style="font-size: 14px;">Piecewise Functions</span>

<span style="font-size: 14px;">Many activation functions are defined **piecewise**, meaning they use different formulas on different regions of the input domain. For a piecewise function, the "interesting" points to check are the **boundaries** between the pieces, where the formula switches.</span>

<span style="font-size: 14px;">At a boundary point $x = a$:</span>

* <span style="font-size: 14px;">**Continuity** holds if both pieces give the same value at $a$</span>
* <span style="font-size: 14px;">**Differentiability** holds if both pieces give the same derivative at $a$</span>

<span style="font-size: 14px;">Away from boundaries, each piece is typically a smooth function (polynomial, exponential, etc.), so continuity and differentiability are automatic.</span>

## <span style="font-size: 14px;">ReLU (Rectified Linear Unit)</span>

<span style="font-size: 14px;">The most widely used activation function in deep learning:</span>

$$
\text{ReLU}(x) = \max(0, x) = \begin{cases} x & \text{if } x \geq 0 \\ 0 & \text{if } x < 0 \end{cases}
$$

### <span style="font-size: 14px;">Continuity Analysis</span>

<span style="font-size: 14px;">At the boundary $x = 0$:</span>

* <span style="font-size: 14px;">Left limit: $\lim_{x \to 0^-} 0 = 0$</span>
* <span style="font-size: 14px;">Right limit: $\lim_{x \to 0^+} x = 0$</span>
* <span style="font-size: 14px;">Function value: $\text{ReLU}(0) = 0$</span>

<span style="font-size: 14px;">All three agree, so ReLU is **continuous at $x = 0$** and therefore continuous everywhere.</span>

### <span style="font-size: 14px;">Differentiability Analysis</span>

<span style="font-size: 14px;">At $x = 0$:</span>

* <span style="font-size: 14px;">Left derivative: $f'_-(0) = \lim_{h \to 0^+} \frac{0 - 0}{h} = 0$</span>
* <span style="font-size: 14px;">Right derivative: $f'_+(0) = \lim_{h \to 0^+} \frac{h - 0}{h} = 1$</span>

<span style="font-size: 14px;">Since $f'_-(0) = 0 \neq 1 = f'_+(0)$, ReLU is **not differentiable at $x = 0$**. The graph has a sharp "kink" at the origin where the slope abruptly changes from 0 to 1.</span>

<span style="font-size: 14px;">At all other points, the derivative is well-defined:</span>

$$
\text{ReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x < 0 \\ \text{undefined} & \text{if } x = 0 \end{cases}
$$

### <span style="font-size: 14px;">Practical Impact</span>

<span style="font-size: 14px;">Despite the non-differentiability at $x = 0$, ReLU works well in practice because:</span>

* <span style="font-size: 14px;">The probability of any neuron receiving exactly $x = 0$ is essentially zero for continuous-valued inputs</span>
* <span style="font-size: 14px;">In implementation, a **subgradient convention** is used: $\text{ReLU}'(0)$ is set to either 0 or 1 (frameworks typically use 0)</span>
* <span style="font-size: 14px;">The piecewise linear nature makes the gradient either 0 (dead) or 1 (alive), which helps avoid the vanishing gradient problem</span>

<span style="font-size: 14px;">However, ReLU has a problem: the **dying ReLU** phenomenon. If a neuron's input is always negative, its gradient is always 0, and the neuron "dies" (never updates). This motivated the development of Leaky ReLU.</span>

## <span style="font-size: 14px;">Leaky ReLU</span>

<span style="font-size: 14px;">A modification of ReLU that allows a small gradient for negative inputs:</span>

$$
\text{LeakyReLU}(x) = \begin{cases} x & \text{if } x \geq 0 \\ \alpha x & \text{if } x < 0 \end{cases}
$$

<span style="font-size: 14px;">where $\alpha$ is a small positive constant (e.g., a value much less than 1).</span>

### <span style="font-size: 14px;">Continuity Analysis</span>

<span style="font-size: 14px;">At $x = 0$:</span>

* <span style="font-size: 14px;">Left limit: $\lim_{x \to 0^-} \alpha x = 0$</span>
* <span style="font-size: 14px;">Right limit: $\lim_{x \to 0^+} x = 0$</span>
* <span style="font-size: 14px;">Function value: $\text{LeakyReLU}(0) = 0$</span>

<span style="font-size: 14px;">All three agree, so Leaky ReLU is **continuous everywhere**.</span>

### <span style="font-size: 14px;">Differentiability Analysis</span>

<span style="font-size: 14px;">At $x = 0$:</span>

* <span style="font-size: 14px;">Left derivative: $f'_-(0) = \alpha$</span>
* <span style="font-size: 14px;">Right derivative: $f'_+(0) = 1$</span>

<span style="font-size: 14px;">Since $\alpha \neq 1$, Leaky ReLU is **not differentiable at $x = 0$**. There is still a kink, but the slope changes from $\alpha$ to 1 instead of from 0 to 1.</span>

$$
\text{LeakyReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ \alpha & \text{if } x < 0 \\ \text{undefined} & \text{if } x = 0 \end{cases}
$$

### <span style="font-size: 14px;">Practical Impact</span>

<span style="font-size: 14px;">Leaky ReLU solves the dying ReLU problem because the gradient is never exactly zero: it is $\alpha$ for negative inputs. However, it still has the kink at $x = 0$, which means the gradient is not a smooth function of the input. This can cause issues in optimization methods that rely on higher-order derivative information (e.g., Newton's method, natural gradient).</span>

## <span style="font-size: 14px;">GELU (Gaussian Error Linear Unit)</span>

<span style="font-size: 14px;">GELU was introduced by Hendrycks and Gimpel in 2016 and has become the default activation in transformer architectures (BERT, GPT, etc.):</span>

$$
\text{GELU}(x) = x \cdot \Phi(x)
$$

<span style="font-size: 14px;">where $\Phi(x)$ is the standard normal cumulative distribution function (CDF). A commonly used approximation is:</span>

$$
\begin{aligned}
\text{GELU}(x) &\approx 0.5 \, x \left(1 + \tanh\!\left(\sqrt{\frac{2}{\pi}} \left(x + 0.044715 \, x^3\right)\right)\right)
\end{aligned}
$$

### <span style="font-size: 14px;">Continuity Analysis</span>

<span style="font-size: 14px;">GELU is a product of two continuous functions ($x$ and $\Phi(x)$), so it is **continuous everywhere**. There is no piecewise definition, no boundary to worry about.</span>

### <span style="font-size: 14px;">Differentiability Analysis</span>

<span style="font-size: 14px;">Since $\Phi(x)$ is infinitely differentiable (it is the integral of the Gaussian density, which is smooth), and $x$ is a polynomial (hence smooth), their product $x \cdot \Phi(x)$ is also **infinitely differentiable**. GELU has no kinks, no corners, no points where the derivative fails to exist.</span>

<span style="font-size: 14px;">The derivative of GELU is:</span>

$$
\text{GELU}'(x) = \Phi(x) + x \cdot \phi(x)
$$

<span style="font-size: 14px;">where $\phi(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2}$ is the standard normal probability density function (PDF). This derivative is a smooth, well-behaved function for all $x$.</span>

### <span style="font-size: 14px;">Practical Impact</span>

<span style="font-size: 14px;">GELU's smoothness offers several advantages:</span>

* <span style="font-size: 14px;">**Smooth gradients**: no abrupt changes in the gradient, which can improve optimization dynamics</span>
* <span style="font-size: 14px;">**Probabilistic interpretation**: GELU can be seen as a stochastic regularizer that randomly zeros out inputs based on how negative they are</span>
* <span style="font-size: 14px;">**Better second-order behavior**: methods that use curvature information benefit from the smoothness of GELU</span>
* <span style="font-size: 14px;">**Non-monotonic**: unlike ReLU, GELU slightly dips below zero for negative inputs before approaching zero, which can provide useful signal</span>

## <span style="font-size: 14px;">Comparison of Differentiability Properties</span>

<span style="font-size: 14px;">Summarizing the three activations:</span>

* <span style="font-size: 14px;">**ReLU**: continuous everywhere, not differentiable at $x = 0$ (left derivative $= 0$, right derivative $= 1$)</span>
* <span style="font-size: 14px;">**Leaky ReLU**: continuous everywhere, not differentiable at $x = 0$ (left derivative $= \alpha$, right derivative $= 1$)</span>
* <span style="font-size: 14px;">**GELU**: continuous and differentiable everywhere (infinitely smooth)</span>

<span style="font-size: 14px;">The progression from ReLU to Leaky ReLU to GELU shows an evolution toward smoother activation functions. Each step addresses a limitation of the previous one:</span>

* <span style="font-size: 14px;">ReLU solves the vanishing gradient problem of sigmoid/tanh but introduces dead neurons</span>
* <span style="font-size: 14px;">Leaky ReLU solves dead neurons but retains the kink</span>
* <span style="font-size: 14px;">GELU removes the kink entirely while maintaining the desirable property of suppressing negative inputs</span>

## <span style="font-size: 14px;">Why Smoothness Matters in Deep Learning</span>

<span style="font-size: 14px;">The differentiability of activation functions has concrete implications for training neural networks:</span>

### <span style="font-size: 14px;">Gradient-Based Optimization</span>

<span style="font-size: 14px;">Backpropagation computes gradients by applying the chain rule through every layer. At a non-differentiable point, the chain rule technically does not apply. In practice, frameworks handle this by using subgradients (choosing either the left or right derivative at the kink), but this introduces a mathematical inconsistency that can affect convergence proofs.</span>

### <span style="font-size: 14px;">Loss Landscape Smoothness</span>

<span style="font-size: 14px;">Non-differentiable activations create "ridges" in the loss landscape where the gradient changes direction abruptly. This can slow down optimization, especially for methods that assume smooth loss functions (Adam, L-BFGS, natural gradient). Smooth activations like GELU produce smoother loss landscapes that are easier to optimize.</span>

### <span style="font-size: 14px;">Higher-Order Methods</span>

<span style="font-size: 14px;">Some advanced optimization techniques use second derivatives (Hessian) or higher-order information. At a kink, the second derivative does not exist, which limits the applicability of these methods. Smooth activations allow the use of the full calculus toolkit.</span>

### <span style="font-size: 14px;">Numerical Stability</span>

<span style="font-size: 14px;">At non-differentiable points, the gradient depends on the direction of approach. With floating-point arithmetic, whether a value is exactly 0.0 or a tiny positive/negative number is somewhat arbitrary, leading to unpredictable behavior at kinks. Smooth functions avoid this issue entirely.</span>

## <span style="font-size: 14px;">Numerical Derivative Computation</span>

<span style="font-size: 14px;">To numerically verify differentiability at a point $x = a$, we compute one-sided finite differences using a small step $h$:</span>

$$
f'_-(a) \approx \frac{f(a) - f(a - h)}{h}
$$

$$
f'_+(a) \approx \frac{f(a + h) - f(a)}{h}
$$

<span style="font-size: 14px;">These are **one-sided difference quotients**. If their values agree (within numerical tolerance), the function is differentiable at $a$. If they disagree, there is a kink.</span>

<span style="font-size: 14px;">The choice of $h$ involves a tradeoff:</span>

* <span style="font-size: 14px;">**Too large $h$**: the finite difference is a poor approximation of the true derivative</span>
* <span style="font-size: 14px;">**Too small $h$**: floating-point cancellation errors dominate (subtracting two nearly equal numbers)</span>

<span style="font-size: 14px;">The sweet spot for $h$ balances approximation error and numerical precision: small enough that the finite difference closely approximates the derivative, but large enough to avoid floating-point cancellation in 64-bit arithmetic.</span>

## <span style="font-size: 14px;">The Subgradient Convention</span>

<span style="font-size: 14px;">At a non-differentiable point, deep learning frameworks need to return some gradient value for backpropagation to proceed. The standard approach uses a **subgradient**: any value between the left and right derivatives.</span>

<span style="font-size: 14px;">For ReLU at $x = 0$:</span>

* <span style="font-size: 14px;">Left derivative: 0</span>
* <span style="font-size: 14px;">Right derivative: 1</span>
* <span style="font-size: 14px;">Valid subgradients: any value in $[0, 1]$</span>
* <span style="font-size: 14px;">Common convention: use 0 (PyTorch default)</span>

<span style="font-size: 14px;">For Leaky ReLU at $x = 0$:</span>

* <span style="font-size: 14px;">Left derivative: $\alpha$</span>
* <span style="font-size: 14px;">Right derivative: 1</span>
* <span style="font-size: 14px;">Valid subgradients: any value in $[\alpha, 1]$</span>
* <span style="font-size: 14px;">Common convention: use $\alpha$</span>

<span style="font-size: 14px;">The choice of subgradient at a single point has negligible effect on training because the probability of any neuron receiving exactly $x = 0$ is essentially zero for continuous input distributions.</span>

## <span style="font-size: 14px;">Other Smooth Activations</span>

<span style="font-size: 14px;">GELU is not the only smooth alternative to ReLU. Other smooth activations include:</span>

* <span style="font-size: 14px;">**Swish/SiLU**: $f(x) = x \cdot \sigma(x)$ where $\sigma$ is the sigmoid function. Smooth everywhere, self-gated.</span>
* <span style="font-size: 14px;">**Softplus**: $f(x) = \ln(1 + e^x)$. A smooth approximation of ReLU. Approaches ReLU as $x \to \pm\infty$.</span>
* <span style="font-size: 14px;">**Mish**: $f(x) = x \cdot \tanh(\text{softplus}(x))$. Smooth and non-monotonic.</span>
* <span style="font-size: 14px;">**ELU**: $f(x) = x$ if $x \geq 0$, $\alpha(e^x - 1)$ if $x < 0$. Continuous and differentiable everywhere (including at $x = 0$, since both pieces have derivative 1 there when $\alpha = 1$).</span>

<span style="font-size: 14px;">The trend in modern architectures is toward smooth activations, driven by both empirical performance gains and theoretical advantages in optimization.</span>

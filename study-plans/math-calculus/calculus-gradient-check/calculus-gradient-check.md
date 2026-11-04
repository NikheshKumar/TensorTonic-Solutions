<span style="font-size: 14px;">Every time you implement a custom backward pass, write a new loss function, or modify a neural network layer, you face a question: is my gradient computation correct? A single sign error or missing term in the gradient will not cause an immediate crash - the model will simply train poorly, and the bug can be invisible for days. Finite difference gradient checking is the systematic answer to this problem. It computes a numerical approximation of the gradient using only function evaluations, providing an independent reference to verify analytical gradients.</span>

## <span style="font-size: 14px;">The Derivative as a Limit</span>

<span style="font-size: 14px;">The definition of the derivative of $f$ at a point $x$ is:</span>

$$
f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}
$$

<span style="font-size: 14px;">This limit involves an infinitesimally small $h$, which we cannot represent on a computer. Instead, we choose a small but finite $h$ and compute the ratio. The question is: how does the approximation error depend on $h$, and what is the best choice of $h$?</span>

<span style="font-size: 14px;">The answer involves two competing sources of error. Making $h$ smaller reduces the approximation error from truncation (the mathematical error from using a finite $h$). But making $h$ too small amplifies the error from floating-point roundoff (the computational error from finite precision). The optimal $h$ balances these two forces.</span>

## <span style="font-size: 14px;">Forward Difference Approximation</span>

<span style="font-size: 14px;">The simplest finite difference approximation is the forward difference:</span>

$$
f'(x) \approx \frac{f(x + h) - f(x)}{h}
$$

<span style="font-size: 14px;">To analyze its accuracy, we use Taylor's theorem. Expanding $f(x + h)$ around $x$:</span>

$$
f(x + h) = f(x) + f'(x) h + \frac{f''(x)}{2} h^2 + \frac{f'''(x)}{6} h^3 + \cdots
$$

<span style="font-size: 14px;">Substituting into the forward difference formula:</span>

$$
\frac{f(x + h) - f(x)}{h} = \frac{f'(x) h + \frac{f''(x)}{2} h^2 + \cdots}{h} = f'(x) + \frac{f''(x)}{2} h + O(h^2)
$$

<span style="font-size: 14px;">The error is:</span>

$$
\text{Error}_{\text{forward}} = \frac{f''(x)}{2} h + O(h^2)
$$

<span style="font-size: 14px;">This is called a first-order approximation because the leading error term is proportional to $h^1$. Halving $h$ approximately halves the error. The error is biased - it consistently over- or under-estimates the derivative depending on the sign of $f''(x)$ (the curvature).</span>

## <span style="font-size: 14px;">Backward Difference Approximation</span>

<span style="font-size: 14px;">The backward difference uses the point behind $x$ instead of ahead:</span>

$$
f'(x) \approx \frac{f(x) - f(x - h)}{h}
$$

<span style="font-size: 14px;">Expanding $f(x - h)$:</span>

$$
f(x - h) = f(x) - f'(x) h + \frac{f''(x)}{2} h^2 - \frac{f'''(x)}{6} h^3 + \cdots
$$

<span style="font-size: 14px;">Substituting:</span>

$$
\frac{f(x) - f(x - h)}{h} = f'(x) - \frac{f''(x)}{2} h + O(h^2)
$$

<span style="font-size: 14px;">The backward difference also has first-order error, but with the opposite sign on the leading term. If the forward difference overestimates, the backward difference underestimates by approximately the same amount. This observation leads directly to the central difference.</span>

## <span style="font-size: 14px;">Central Difference Approximation</span>

<span style="font-size: 14px;">The central difference averages the forward and backward formulas:</span>

$$
f'(x) \approx \frac{f(x + h) - f(x - h)}{2h}
$$

<span style="font-size: 14px;">Subtracting the Taylor expansions:</span>

$$
f(x + h) - f(x - h) = 2f'(x) h + \frac{f'''(x)}{3} h^3 + O(h^5)
$$

<span style="font-size: 14px;">Dividing by $2h$:</span>

$$
\frac{f(x + h) - f(x - h)}{2h} = f'(x) + \frac{f'''(x)}{6} h^2 + O(h^4)
$$

<span style="font-size: 14px;">The first-order error term has cancelled out! The central difference has second-order accuracy:</span>

$$
\text{Error}_{\text{central}} = \frac{f'''(x)}{6} h^2 + O(h^4)
$$

<span style="font-size: 14px;">Halving $h$ now reduces the error by a factor of 4 (not 2). For the same $h$, the central difference is dramatically more accurate than the forward or backward difference. This is why the central difference is the standard choice for gradient checking.</span>

### <span style="font-size: 14px;">Why the Cancellation Happens</span>

<span style="font-size: 14px;">The cancellation of the $O(h)$ error term is not a coincidence. The forward difference has error $+\frac{f''}{2}h$ and the backward difference has error $-\frac{f''}{2}h$. Their average has error $0 \cdot h + O(h^2)$. This is an instance of Richardson extrapolation - combining two approximations of different biases to get a higher-order result. The same principle underlies Romberg integration and other numerical methods.</span>

## <span style="font-size: 14px;">Floating-Point Roundoff Error</span>

<span style="font-size: 14px;">On a computer, real numbers are stored in IEEE 754 floating-point format with finite precision. For 64-bit doubles, the machine epsilon is $\epsilon_{\text{mach}} \approx 2.2 \times 10^{-16}$, meaning any real number $a$ is stored as $\text{fl}(a) = a(1 + \delta)$ where $|\delta| \leq \epsilon_{\text{mach}}$.</span>

<span style="font-size: 14px;">When we compute $f(x + h)$, the result has a roundoff error of approximately $\epsilon_{\text{mach}} \cdot |f(x)|$. In the forward difference $(f(x+h) - f(x))/h$, we subtract two nearly equal numbers (catastrophic cancellation), amplifying the roundoff error:</span>

$$
\text{Roundoff error} \approx \frac{2\epsilon_{\text{mach}} |f(x)|}{h}
$$

<span style="font-size: 14px;">As $h$ decreases, this roundoff error grows like $1/h$. Eventually, for very small $h$, the roundoff dominates and the total error starts increasing.</span>

## <span style="font-size: 14px;">Optimal Step Size</span>

<span style="font-size: 14px;">The total error is the sum of truncation error and roundoff error:</span>

### <span style="font-size: 14px;">Forward/Backward Difference</span>

$$
E_{\text{total}} \approx \frac{|f''(x)|}{2} h + \frac{2\epsilon_{\text{mach}} |f(x)|}{h}
$$

<span style="font-size: 14px;">Minimizing by taking $dE/dh = 0$:</span>

$$
\begin{aligned}
\frac{|f''(x)|}{2} &= \frac{2\epsilon_{\text{mach}} |f(x)|}{h^2} \\
&\implies h^* = 2\sqrt{\frac{\epsilon_{\text{mach}} |f(x)|}{|f''(x)|}}
\end{aligned}
$$

<span style="font-size: 14px;">For typical functions where $|f(x)|$ and $|f''(x)|$ are order 1, this gives $h^* \sim \sqrt{\epsilon_{\text{mach}}}$. The minimum error at this optimal $h$ scales as $\sqrt{\epsilon_{\text{mach}}}$, which limits the number of accurate digits the forward difference can achieve.</span>

### <span style="font-size: 14px;">Central Difference</span>

$$
E_{\text{total}} \approx \frac{|f'''(x)|}{6} h^2 + \frac{2\epsilon_{\text{mach}} |f(x)|}{h}
$$

<span style="font-size: 14px;">Minimizing:</span>

$$
\begin{aligned}
\frac{|f'''(x)|}{3} h &= \frac{2\epsilon_{\text{mach}} |f(x)|}{h^2} \\
&\implies h^* = \left(\frac{6\epsilon_{\text{mach}} |f(x)|}{|f'''(x)|}\right)^{1/3}
\end{aligned}
$$

<span style="font-size: 14px;">For order-1 functions, $h^* \approx (6\epsilon_{\text{mach}})^{1/3} \approx 10^{-5}$. The minimum error is proportional to $\epsilon_{\text{mach}}^{2/3} \approx 10^{-11}$, so central differences can achieve about 11 digits of accuracy - three more than forward differences.</span>

## <span style="font-size: 14px;">Error Behavior Across h Values</span>

<span style="font-size: 14px;">When you plot $\log(\text{error})$ vs. $\log(h)$ for a range of $h$ values, you see a characteristic V-shape (or U-shape on a log-log plot):</span>

<span style="font-size: 14px;">1. For large $h$ (left side), truncation error dominates. The error decreases as $h$ shrinks, with slope 1 for forward/backward and slope 2 for central on a log-log plot.</span>

<span style="font-size: 14px;">2. At the optimal $h$ (bottom of the V), the two error sources are balanced.</span>

<span style="font-size: 14px;">3. For small $h$ (right side), roundoff error dominates. The error increases as $h$ shrinks, with slope $-1$ on a log-log plot.</span>

<span style="font-size: 14px;">This behavior is universal: it occurs for any function, any point, and any finite difference formula. The only things that change are the location of the optimum and the minimum error level.</span>

### <span style="font-size: 14px;">Interpreting the V-Shape on a Log-Log Plot</span>

<span style="font-size: 14px;">The V-shape reveals a deep principle about numerical computation: there is a fundamental accuracy ceiling imposed by the representation of numbers in finite precision. No amount of algorithmic cleverness can push past this ceiling for a given arithmetic precision. The left branch of the V is governed by mathematical approximation theory (Taylor series), while the right branch is governed by the hardware representation of numbers. The optimal $h$ sits at their intersection, and the minimum error at that point represents the best that a given finite difference formula can achieve.</span>

<span style="font-size: 14px;">Comparing the V-shapes of the forward and central difference on the same plot is instructive. The central difference V-shape has a steeper left branch (slope 2 vs slope 1) and a lower minimum. The right branches have the same slope (-1), since both formulas suffer from the same roundoff mechanism. The practical consequence is that the central difference achieves several additional digits of accuracy at its optimal $h$ compared to the forward difference at its own optimal $h$.</span>

### <span style="font-size: 14px;">The Geometry of Error Cancellation</span>

<span style="font-size: 14px;">The superiority of the central difference can also be understood geometrically. The forward difference computes the slope of a secant line from $(x, f(x))$ to $(x+h, f(x+h))$. This secant is anchored at one end of the interval, so its slope is biased by the curvature of $f$. The central difference computes the slope of a secant from $(x-h, f(x-h))$ to $(x+h, f(x+h))$, which is centered at $x$. By symmetry, the curvature-induced bias on the left half of the interval approximately cancels the bias on the right half. This geometric cancellation is the visual counterpart of the algebraic cancellation in the Taylor series.</span>

<span style="font-size: 14px;">More precisely, the tangent line at $x$ has slope $f'(x)$. The secant from $x$ to $x+h$ lies above or below the tangent depending on the sign of $f''(x)$, while the secant from $x-h$ to $x$ lies on the opposite side. Averaging these two secants (which is what the central difference does) produces a secant whose slope is much closer to the true tangent slope. The remaining error comes from the third derivative, which measures how the curvature itself is changing - a subtler and smaller effect.</span>

## <span style="font-size: 14px;">Application: torch.autograd.gradcheck</span>

<span style="font-size: 14px;">PyTorch's `torch.autograd.gradcheck` function uses the central difference formula to verify that the analytical gradient computed by autograd matches a numerical approximation. It evaluates the function at $x + h \cdot e_i$ and $x - h \cdot e_i$ for each coordinate direction $e_i$ and compares the result to the autograd gradient.</span>

<span style="font-size: 14px;">The default step size is $h = 10^{-6}$, which is close to the theoretical optimum for central differences. The default tolerance is $10^{-5}$ for the absolute difference and $10^{-3}$ for the relative difference. These tolerances account for the fact that numerical gradients are not exact - they have inherent error from both truncation and roundoff.</span>

<span style="font-size: 14px;">If `gradcheck` fails, it usually means either: (a) the analytical gradient implementation has a bug, or (b) the function involves operations where numerical gradients are unreliable (like very sharp activations, or operations near singularities). In case (a), the fix is to correct the gradient. In case (b), the fix is to increase $h$ or use a more sophisticated numerical method.</span>

## <span style="font-size: 14px;">Gradient Checking in Practice</span>

### <span style="font-size: 14px;">The Relative Error Criterion</span>

<span style="font-size: 14px;">When comparing analytical and numerical gradients, the absolute error $|g_{\text{analytical}} - g_{\text{numerical}}|$ is not always the right metric. If the gradient itself is large (say, 1000), an absolute error of 0.01 is excellent. If the gradient is tiny (say, $10^{-6}$), an absolute error of 0.01 is terrible.</span>

<span style="font-size: 14px;">The standard practice is to use a relative error criterion:</span>

$$
\begin{aligned}
\text{relative error} &= \frac{|g_{\text{analytical}} - g_{\text{numerical}}|}{\max(|g_{\text{analytical}}|, |g_{\text{numerical}}|, \epsilon)}
\end{aligned}
$$

<span style="font-size: 14px;">where $\epsilon$ is a small constant to prevent division by zero. A relative error below $10^{-5}$ is generally considered passing, below $10^{-7}$ is excellent, and above $10^{-2}$ indicates a likely bug.</span>

### <span style="font-size: 14px;">Common Pitfalls</span>

<span style="font-size: 14px;">Several situations can cause gradient checking to give misleading results:</span>

<span style="font-size: 14px;">1. Non-differentiable functions: ReLU, absolute value, and max operations have kinks where the numerical gradient may disagree with the (sub)gradient convention used by the framework.</span>

<span style="font-size: 14px;">2. Stochastic operations: Dropout, stochastic depth, and other random operations produce different outputs for $f(x+h)$ and $f(x-h)$, making the numerical gradient meaningless. Always disable randomness during gradient checking.</span>

<span style="font-size: 14px;">3. Very large or very small values: Functions with extreme outputs can cause overflow in the numerator or underflow in the denominator of the finite difference.</span>

<span style="font-size: 14px;">4. Ill-conditioned functions: Near singularities or sharp transitions, even the central difference can have large errors.</span>

### <span style="font-size: 14px;">Multi-Dimensional Gradient Checking</span>

<span style="font-size: 14px;">For a function $f: \mathbb{R}^n \to \mathbb{R}$, gradient checking verifies each partial derivative independently:</span>

$$
\frac{\partial f}{\partial x_i} \approx \frac{f(x + h e_i) - f(x - h e_i)}{2h}
$$

<span style="font-size: 14px;">where $e_i$ is the $i$-th standard basis vector. This requires $2n$ function evaluations (two per parameter), which is why gradient checking is expensive for models with millions of parameters. In practice, gradient checking is performed on a small subset of parameters or on a simplified version of the model.</span>

## <span style="font-size: 14px;">Higher-Order Finite Differences</span>

<span style="font-size: 14px;">The central difference can be extended to higher accuracy by using more function evaluations. The five-point stencil:</span>

$$
f'(x) \approx \frac{-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)}{12h}
$$

<span style="font-size: 14px;">has fourth-order accuracy: $\text{Error} = O(h^4)$. This achieves higher accuracy for moderate $h$ but does not fundamentally change the roundoff barrier - the optimal error is still limited by machine epsilon.</span>

<span style="font-size: 14px;">For second derivatives, the standard central difference is:</span>

$$
f''(x) \approx \frac{f(x+h) - 2f(x) + f(x-h)}{h^2}
$$

<span style="font-size: 14px;">This has second-order accuracy in $h$ and is used in optimization for approximating the Hessian when analytical second derivatives are unavailable.</span>

## <span style="font-size: 14px;">Summary of Error Orders</span>

<span style="font-size: 14px;">For reference, the key results for each finite difference method:</span>

<span style="font-size: 14px;">Forward difference: truncation error $O(h)$, optimal $h \sim \sqrt{\epsilon_{\text{mach}}} \approx 10^{-8}$, best achievable error $\sim 10^{-8}$</span>

<span style="font-size: 14px;">Backward difference: truncation error $O(h)$, same optimal $h$ and error as forward</span>

<span style="font-size: 14px;">Central difference: truncation error $O(h^2)$, optimal $h \sim \epsilon_{\text{mach}}^{1/3} \approx 10^{-5}$, best achievable error $\sim 10^{-11}$</span>

<span style="font-size: 14px;">These results explain why $h = 10^{-5}$ or $10^{-6}$ is the standard default for gradient checking: it is near-optimal for central differences, which is the method of choice. Using a much smaller $h$ (like $10^{-10}$) would actually produce worse results due to roundoff, which is counterintuitive but fundamental to numerical computation.</span>
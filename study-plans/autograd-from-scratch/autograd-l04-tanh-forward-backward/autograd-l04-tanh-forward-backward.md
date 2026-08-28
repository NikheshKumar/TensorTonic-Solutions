# <span style="font-size: 20px;">Differentiate a Tanh Activation</span>

<span style="font-size: 14px;">The hyperbolic tangent transforms any finite scalar into a value between $-1$ and $1$. This problem performs its forward calculation, then manually applies the chain rule to determine how an incoming gradient affects the activation's input.</span>

$$
y = \tanh(x)
$$

---

## <span style="font-size: 16px;">What tanh does</span>

<span style="font-size: 14px;">Near zero, tanh responds strongly to changes in its input. For large positive inputs it approaches $1$, and for large negative inputs it approaches $-1$. Its smooth S-shaped curve makes it useful for seeing how activation values and activation gradients are connected.</span>

<span style="font-size: 14px;">The forward output should be computed once and retained because the derivative can be expressed directly in terms of that saved value.</span>

---

## <span style="font-size: 16px;">The local derivative</span>

<span style="font-size: 14px;">The derivative of tanh at its input is:</span>

$$
\frac{dy}{dx} = 1 - \tanh^2(x)
$$

<span style="font-size: 14px;">Since $y$ already stores the tanh output, the same derivative becomes:</span>

$$
\frac{dy}{dx} = 1-y^2
$$

<span style="font-size: 14px;">This saved-output form avoids evaluating tanh a second time and mirrors how autograd nodes retain forward values for use during backward propagation.</span>

---

## <span style="font-size: 16px;">Combining with the upstream gradient</span>

<span style="font-size: 14px;">The activation is usually part of a larger graph. Let $g$ be the derivative of the final loss with respect to the tanh output:</span>

$$
g = \frac{\partial L}{\partial y}
$$

<span style="font-size: 14px;">The chain rule multiplies this upstream influence by the local derivative:</span>

$$
\frac{\partial L}{\partial x}
=
\frac{\partial L}{\partial y}
\frac{\partial y}{\partial x}
=
g(1-y^2)
$$

<span style="font-size: 14px;">The upstream gradient controls the sign and scale of the result. The local derivative controls how much of that signal can pass through tanh at the chosen input.</span>

---

## <span style="font-size: 16px;">A numerical example</span>

<span style="font-size: 14px;">Consider an input of $0.5$ and an upstream gradient of $2$. The forward output is approximately:</span>

$$
y = \tanh(0.5) \approx 0.462117
$$

<span style="font-size: 14px;">The local derivative is:</span>

$$
1-y^2 \approx 1-(0.462117)^2 \approx 0.786448
$$

<span style="font-size: 14px;">After applying the upstream gradient:</span>

$$
\frac{\partial L}{\partial x}
\approx
2(0.786448)
\approx
1.572895
$$

<span style="font-size: 14px;">At zero, tanh also equals zero, so its local derivative is one and the upstream gradient passes through unchanged.</span>

---

## <span style="font-size: 16px;">Saturation and small gradients</span>

<span style="font-size: 14px;">When the input magnitude is large, the output lies very close to $1$ or $-1$. Squaring that output gives a value close to one, so the local derivative becomes close to zero.</span>

$$
|x| \text{ large}
\quad\Longrightarrow\quad
y^2 \approx 1
\quad\Longrightarrow\quad
1-y^2 \approx 0
$$

<span style="font-size: 14px;">This is tanh saturation. Even a substantial upstream gradient can become very small after passing through a saturated activation. Lower-precision dtypes may round the output to its limiting value sooner, which can make the computed derivative exactly zero.</span>

---

## <span style="font-size: 16px;">Tensor contract</span>

<span style="font-size: 14px;">Both inputs are zero-dimensional floating tensors on the same device and with the same dtype. PyTorch tensor operations preserve those properties, so the output and input gradient should remain scalar tensors rather than being converted into Python numbers.</span>

<span style="font-size: 14px;">The derivative must be assembled manually from the saved output and upstream gradient. Calling automatic backward propagation would bypass the exact calculation this problem is designed to practice.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Forgetting the upstream gradient.** The local derivative alone is not the requested loss gradient.</span>
* <span style="font-size: 14px;">**Using the input in place of the output.** The saved-output formula is $1-y^2$, not $1-x^2$.</span>
* <span style="font-size: 14px;">**Changing dtype or device.** Converting through a Python scalar can break the tensor-preservation requirement.</span>
* <span style="font-size: 14px;">**Expecting a large gradient in saturation.** Values near the tanh limits naturally produce a local derivative near zero.</span>

<span style="font-size: 14px;">The complete backward rule is the product of two ideas: tanh contributes the local factor $1-y^2$, and the rest of the graph contributes the upstream factor $g$.</span>

---
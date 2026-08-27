# <span style="font-size: 20px;">Measure Scalar Expression Partials</span>

<span style="font-size: 14px;">A function can depend on several inputs at once. To understand the influence of one input, change that input slightly while keeping every other input fixed. The resulting rate of change is called a partial derivative.</span>

<span style="font-size: 14px;">This problem uses the scalar expression:</span>

$$
d=ab+c
$$

<span style="font-size: 14px;">There are three inputs, so there are three local questions: how does $d$ respond to a small change in $a$, in $b$, or in $c$? Each question must be measured independently.</span>

---

## <span style="font-size: 16px;">Change one input at a time</span>

<span style="font-size: 14px;">First compute the unperturbed value:</span>

$$
d_0=ab+c
$$

<span style="font-size: 14px;">To estimate the partial derivative with respect to $a$, increase only $a$ by the positive step $h$:</span>

$$
\frac{\partial d}{\partial a}\approx
\frac{d(a+h,b,c)-d(a,b,c)}{h}
$$

<span style="font-size: 14px;">The same pattern applies to the other inputs:</span>

$$
\frac{\partial d}{\partial b}\approx
\frac{d(a,b+h,c)-d(a,b,c)}{h}
$$

$$
\frac{\partial d}{\partial c}\approx
\frac{d(a,b,c+h)-d(a,b,c)}{h}
$$

<span style="font-size: 14px;">All three estimates use the same baseline $d(a,b,c)$. The only difference is which input receives the perturbation.</span>

---

## <span style="font-size: 16px;">Why the three sensitivities have different values</span>

<span style="font-size: 14px;">The expression itself reveals what each partial should be. If only $a$ changes, then $b$ tells us how strongly that change is multiplied:</span>

$$
d(a+h,b,c)-d(a,b,c)=(a+h)b+c-(ab+c)=hb
$$

<span style="font-size: 14px;">Dividing by $h$ gives:</span>

$$
\frac{\partial d}{\partial a}=b
$$

<span style="font-size: 14px;">By the same reasoning:</span>

$$
\frac{\partial d}{\partial b}=a
$$

$$
\frac{\partial d}{\partial c}=1
$$

<span style="font-size: 14px;">These exact results are useful for understanding the numerical experiment. The forward differences should be equal or extremely close to $b$, $a$, and $1$. Because this expression is linear in each input when the others are fixed, there is no curvature error in the mathematical quotient. Small discrepancies can still appear because the calculation uses floating-point arithmetic.</span>

---

## <span style="font-size: 16px;">A complete example</span>

<span style="font-size: 14px;">Consider an input where $a$ is $2$, $b$ is $-3$, and $c$ is $10$. Use a forward step of $0.001$. The baseline output is:</span>

$$
d=(2)(-3)+10=4
$$

<span style="font-size: 14px;">Perturbing only $a$ gives:</span>

$$
d(2.001,-3,10)=3.997
$$

$$
\frac{3.997-4}{0.001}=-3
$$

<span style="font-size: 14px;">Perturbing only $b$ gives a partial of $2$, while perturbing only $c$ gives a partial of $1$. In the required order, the expression value and its three partial derivatives are:</span>

$$
\left(d, \frac{\partial d}{\partial a}, \frac{\partial d}{\partial b}, \frac{\partial d}{\partial c}\right) = (4, -3, 2, 1)
$$

<span style="font-size: 14px;">Notice that a negative partial is not an error. Here, increasing $a$ makes the product $ab$ smaller because $b$ is negative.</span>

---

## <span style="font-size: 16px;">Why this matters for autograd</span>

<span style="font-size: 14px;">Automatic differentiation eventually computes these sensitivities without repeatedly perturbing inputs. Before building that machinery, finite differences give us a direct way to ask what each input contributes. They also establish an important habit: when checking the derivative for one variable, every other variable must remain unchanged.</span>

<span style="font-size: 14px;">A clean implementation follows this order:</span>

1. <span style="font-size: 14px;">Convert the four numeric inputs to 64-bit floating-point values.</span>
2. <span style="font-size: 14px;">Evaluate the baseline expression once.</span>
3. <span style="font-size: 14px;">Evaluate three perturbed expressions, changing one input in each call.</span>
4. <span style="font-size: 14px;">Subtract the shared baseline and divide each difference by $h$.</span>
5. <span style="font-size: 14px;">Return the baseline followed by the partials for $a$, $b$, and $c$.</span>

---

## <span style="font-size: 16px;">Common mistakes</span>

- <span style="font-size: 14px;">Perturbing two inputs together measures a combined change, not one partial derivative.</span>
- <span style="font-size: 14px;">Using a perturbed result as the next baseline mixes the three independent experiments.</span>
- <span style="font-size: 14px;">Returning the derivatives in a different order makes correct values appear incorrect.</span>
- <span style="font-size: 14px;">Forgetting that $c$ is added directly can lead to an incorrect derivative for $c$. Its sensitivity is always one.</span>
- <span style="font-size: 14px;">Returning NumPy scalars rather than Python floats does not satisfy the required interface.</span>

<span style="font-size: 14px;">The key idea is isolation. Start from one shared expression value, move one input by $h$, and treat the resulting output change as that input's local contribution.</span>

---
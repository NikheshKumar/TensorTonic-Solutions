# <span style="font-size: 20px;">Estimate a Scalar Derivative</span>

<span style="font-size: 14px;">A derivative answers a simple question: if the input changes by a tiny amount, how quickly does the output change? For a scalar function, this is the slope of the function near the chosen input. A positive derivative means the output is increasing, a negative derivative means it is decreasing, and a derivative near zero means the graph is locally flat.</span>

<span style="font-size: 14px;">In this problem, the function is a polynomial. The coefficient list is arranged from the constant term upward:</span>

$$
f(x) = c_0 + c_1x + c_2x^2 + \cdots + c_{n-1}x^{n-1}
$$

<span style="font-size: 14px;">For example, consider coefficients of $5$ for the constant term, $-4$ for the linear term, and $3$ for the quadratic term. They represent:</span>

$$
f(x) = 5 - 4x + 3x^2
$$

<span style="font-size: 14px;">The order matters. The first number multiplies $x$ to the power zero, the second multiplies $x$ to the power one, and so on.</span>

---

## <span style="font-size: 16px;">From a secant slope to a derivative</span>

<span style="font-size: 14px;">Suppose we evaluate the polynomial at $x$, then move a small distance $h$ to the right and evaluate it again. The output changes by:</span>

$$
f(x+h)-f(x)
$$

<span style="font-size: 14px;">Dividing this output change by the input change gives the slope between the two points:</span>

$$
\frac{f(x+h)-f(x)}{h}
$$

<span style="font-size: 14px;">This is called a forward difference because the second point is ahead of $x$. It is a secant slope, not the exact tangent slope. When $h$ is small, the two points are close together, so the secant slope usually becomes a good estimate of the derivative at $x$.</span>

<span style="font-size: 14px;">The task asks for exactly three values:</span>

- <span style="font-size: 14px;">$f(x)$, the polynomial value at the original point</span>
- <span style="font-size: 14px;">$f(x+h)$, the value after the small forward step</span>
- <span style="font-size: 14px;">the forward-difference slope</span>

---

## <span style="font-size: 16px;">A complete example</span>

<span style="font-size: 14px;">Consider the same quadratic at the point $x=3$, using a forward step of $0.001$. The polynomial is:</span>

$$
f(x)=3x^2-4x+5
$$

<span style="font-size: 14px;">At the original point:</span>

$$
f(3)=3(3^2)-4(3)+5=20
$$

<span style="font-size: 14px;">At the perturbed point:</span>

$$
f(3.001)=20.014003
$$

<span style="font-size: 14px;">The estimated slope is therefore:</span>

$$
\frac{20.014003-20}{0.001}=14.003
$$

<span style="font-size: 14px;">The exact derivative of this polynomial is $6x-4$, which equals $14$ at the chosen input. The small difference between $14.003$ and $14$ comes from measuring a secant over a nonzero interval.</span>

---

## <span style="font-size: 16px;">Evaluating the polynomial correctly</span>

<span style="font-size: 14px;">You could compute every power separately, but Horner's method evaluates the same polynomial with a simple multiply-and-add sequence. Since the input coefficients are stored from lowest power to highest power, Horner's method processes them in reverse order.</span>

<span style="font-size: 14px;">Using those same constant, linear, and quadratic coefficients, the nested form is:</span>

$$
f(x)=(3x-4)x+5
$$

<span style="font-size: 14px;">Starting from the highest coefficient, multiply the running result by the point and add the next coefficient. Repeat this once at $x$ and once at $x+h$. This respects the required coefficient convention and works for constant, linear, and higher-degree polynomials.</span>

---

## <span style="font-size: 16px;">Choosing and interpreting the step</span>

<span style="font-size: 14px;">A smaller positive $h$ usually reduces the gap between the secant slope and the true derivative. However, floating-point numbers have limited precision. If $h$ becomes extremely small, $f(x+h)$ and $f(x)$ can be nearly identical as stored numbers, and their subtraction may lose useful digits. This task supplies $h$, so the implementation should use it directly rather than trying to choose a different method.</span>

<span style="font-size: 14px;">Two useful sanity checks are:</span>

- <span style="font-size: 14px;">A constant polynomial produces the same value at both points, so its estimated derivative is zero.</span>
- <span style="font-size: 14px;">A linear polynomial has the same slope everywhere, so the forward difference recovers its coefficient of $x$.</span>

---

## <span style="font-size: 16px;">Common mistakes</span>

- <span style="font-size: 14px;">Reading the coefficient list as highest power first changes the polynomial itself.</span>
- <span style="font-size: 14px;">Using a central difference solves a related problem, but it does not follow this problem's required forward-difference formula.</span>
- <span style="font-size: 14px;">Dividing by $x+h$ instead of by $h$ confuses the new position with the size of the step.</span>
- <span style="font-size: 14px;">Returning NumPy scalar objects instead of ordinary Python floats violates the output contract.</span>
- <span style="font-size: 14px;">Changing the supplied $h$ can make otherwise reasonable results disagree with the required answers.</span>

<span style="font-size: 14px;">The central idea is to measure one small forward change faithfully. Evaluate the same polynomial at two nearby inputs, subtract the outputs, divide by the known input step, and return the three requested scalar values in order.</span>

---
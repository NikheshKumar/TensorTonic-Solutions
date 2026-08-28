# <span style="font-size: 20px;">Apply a Gradient-Descent Step</span>

<span style="font-size: 14px;">A gradient describes how an objective changes near the current parameters. Each gradient entry corresponds to one parameter. A positive entry says that increasing that parameter would locally increase the objective, while a negative entry says that increasing the parameter would locally decrease it.</span>

<span style="font-size: 14px;">When the goal is to reduce the objective, gradient descent moves in the opposite direction:</span>

$$
\theta'_i=\theta_i-\eta g_i
$$

<span style="font-size: 14px;">Here $\boldsymbol{\theta}$ is the current parameter vector, $\mathbf{g}$ is its gradient, and the nonnegative learning rate $\eta$ controls the step size.</span>

---

## <span style="font-size: 16px;">Reading the update one coordinate at a time</span>

<span style="font-size: 14px;">The rule applies independently to every coordinate:</span>

- <span style="font-size: 14px;">If a gradient is positive, subtracting it makes the parameter smaller.</span>
- <span style="font-size: 14px;">If a gradient is negative, subtracting it makes the parameter larger.</span>
- <span style="font-size: 14px;">If a gradient is zero, that parameter does not move.</span>
- <span style="font-size: 14px;">If the learning rate is zero, none of the parameters move.</span>

<span style="font-size: 14px;">The learning rate scales every movement. It changes how far the update travels, but it does not change the basic direction chosen by the negative gradient.</span>

---

## <span style="font-size: 16px;">A complete update example</span>

<span style="font-size: 14px;">Consider three parameters with current values of $2$, $-3$, and $10$. Their gradients are $-3$, $2$, and $1$, respectively, and the learning rate is $0.1$. Apply the rule to each coordinate:</span>

$$
2-0.1(-3)=2.3
$$

$$
-3-0.1(2)=-3.2
$$

$$
10-0.1(1)=9.9
$$

<span style="font-size: 14px;">The updated parameters are $2.3$, $-3.2$, and $9.9$. The first value rises because its gradient is negative, while the other two fall because their gradients are positive.</span>

---

## <span style="font-size: 16px;">Predicting the local objective change</span>

<span style="font-size: 14px;">The problem also asks for a first-order prediction of how the objective changes. For a small update, the gradient gives the local linear approximation:</span>

$$
\Delta L_{\mathrm{pred}}
=
\sum_i g_i(\theta'_i-\theta_i)
$$

<span style="font-size: 14px;">The quantity in parentheses is the actual update vector. Gradient descent makes that update:</span>

$$
\theta'_i-\theta_i=-\eta g_i
$$

<span style="font-size: 14px;">Substituting it into the prediction gives a useful simplification:</span>

$$
\Delta L_{\mathrm{pred}}
=
-\eta\sum_i g_i^2
$$

<span style="font-size: 14px;">For a nonnegative learning rate, this value cannot be positive. It is zero when the learning rate is zero or when every gradient is zero. Otherwise, the local linear model predicts a decrease.</span>

<span style="font-size: 14px;">In the example:</span>

$$
\Delta L_{\mathrm{pred}}
=
-0.1\left((-3)^2+2^2+1^2\right)
=
-1.4
$$

<span style="font-size: 14px;">The same result comes from pairing the three parameter movements, $0.3$, $-0.2$, and $-0.1$, with their original gradients:</span>

$$
(-3)(0.3)+(2)(-0.2)+(1)(-0.1)=-1.4
$$

---

## <span style="font-size: 16px;">Prediction versus actual change</span>

<span style="font-size: 14px;">The predicted change comes from a local, first-order approximation. It describes what the gradient expects near the current point. The task does not provide an objective function to evaluate again, so it does not ask for the actual change in loss.</span>

<span style="font-size: 14px;">For a sufficiently small step on a smooth objective, the prediction is often informative. For a larger step, curvature can make the real change differ from the linear estimate. The required result should still use the stated formula and the actual update vector.</span>

---

## <span style="font-size: 16px;">Preserving the inputs</span>

<span style="font-size: 14px;">The function must return a fresh list and leave both supplied lists unchanged. This matters because callers may need the original values and gradients for logging, comparison, or another calculation. Converting the inputs into new 64-bit arrays and computing a separate updated array provides a clear non-mutating path.</span>

<span style="font-size: 14px;">A dependable implementation order is:</span>

1. <span style="font-size: 14px;">Convert values and gradients to new 64-bit arrays.</span>
2. <span style="font-size: 14px;">Compute the updated array by subtracting the scaled gradient.</span>
3. <span style="font-size: 14px;">Subtract the original array from the updated array to obtain the actual update vector.</span>
4. <span style="font-size: 14px;">Take its dot product with the gradient.</span>
5. <span style="font-size: 14px;">Return a fresh list of Python floats and one Python float.</span>

---

## <span style="font-size: 16px;">Common mistakes</span>

- <span style="font-size: 14px;">Adding the gradient performs local ascent rather than descent.</span>
- <span style="font-size: 14px;">Computing the dot product of the gradient and updated parameters is not the requested prediction. It must use the change $\boldsymbol{\theta}^{\prime}-\boldsymbol{\theta}$.</span>
- <span style="font-size: 14px;">Editing the input list in place violates the non-mutation requirement.</span>
- <span style="font-size: 14px;">Treating the predicted value as the measured loss change gives it more meaning than the first-order approximation supports.</span>
- <span style="font-size: 14px;">Forgetting the empty-vector case can cause a reduction error, even though its valid predicted change is zero.</span>

<span style="font-size: 14px;">The essential connection is straightforward: the gradient supplies a direction of local increase, the update moves against it, and their dot product predicts how that movement changes the objective nearby.</span>

---
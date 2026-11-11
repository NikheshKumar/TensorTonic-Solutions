# <span style="font-size: 20px;">Build a Scalar Neuron Module</span>

<span style="font-size: 14px;">A scalar neuron module packages a weighted sum, a bias, and an optional nonlinearity into one reusable computation. The same supplied parameters can act as a linear neuron or as a tanh neuron depending on the requested mode.</span>

---

## <span style="font-size: 16px;">The preactivation</span>

<span style="font-size: 14px;">Each input is paired with one weight. Their products are summed, then the scalar bias is added:</span>

$$
a=b+\sum_{i=1}^{n}w_ix_i
$$

<span style="font-size: 14px;">The preactivation $a$ is the neuron's raw scalar response. Weights control the direction and strength of individual input contributions, while the bias shifts the total independently of the input values.</span>

---

## <span style="font-size: 16px;">Linear and nonlinear modes</span>

<span style="font-size: 14px;">Linear mode returns the preactivation directly:</span>

$$
o=a
$$

<span style="font-size: 14px;">Nonlinear mode applies tanh:</span>

$$
o=\tanh(a)
$$

<span style="font-size: 14px;">The mode changes only the final transformation. The weighted sum and bias calculation are identical in both cases.</span>

---

## <span style="font-size: 16px;">Why the optional nonlinearity matters</span>

<span style="font-size: 14px;">A linear neuron can scale and combine its inputs, but stacking linear transformations without nonlinear activations still produces another linear transformation. Tanh changes that behavior by introducing a curved response bounded between $-1$ and $1$.</span>

<span style="font-size: 14px;">Some network outputs intentionally remain linear, especially when an unrestricted scalar is needed. Hidden neurons commonly use a nonlinearity so a larger network can represent relationships beyond weighted sums.</span>

---

## <span style="font-size: 16px;">A linear example</span>

<span style="font-size: 14px;">Consider two inputs with values $1$ and $2$. Their weights are $0.5$ and $-1$, and the bias is $0.25$.</span>

<span style="font-size: 14px;">The weighted contributions are:</span>

$$
(1)(0.5)=0.5
$$

$$
(2)(-1)=-2
$$

<span style="font-size: 14px;">The preactivation is:</span>

$$
a=0.5-2+0.25=-1.25
$$

<span style="font-size: 14px;">In linear mode, the returned scalar is $-1.25$. In nonlinear mode, the same parameters would return $\tanh(-1.25)$, which is approximately $-0.848284$.</span>

---

## <span style="font-size: 16px;">Aligned tensors and scalar output</span>

<span style="font-size: 14px;">The input and weight tensors are one-dimensional with equal shape. Position $i$ in the weight tensor always belongs to position $i$ in the input tensor. Elementwise multiplication forms the contributions, and summation reduces them to a scalar.</span>

<span style="font-size: 14px;">The bias is already a scalar tensor. Adding it to the summed contributions keeps the result zero-dimensional, and tanh preserves that scalar shape when nonlinear mode is active.</span>

---

## <span style="font-size: 16px;">The empty-input case</span>

<span style="font-size: 14px;">Equal empty input and weight tensors contain no weighted contributions. Their sum is zero, so the bias becomes the entire preactivation:</span>

$$
a=b
$$

<span style="font-size: 14px;">With a bias of $1.5$, linear mode returns $1.5$, while nonlinear mode returns:</span>

$$
\tanh(1.5)\approx 0.905148
$$

<span style="font-size: 14px;">This is a valid neuron with zero input width rather than an error case.</span>

---

## <span style="font-size: 16px;">Supplied parameters remain unchanged</span>

<span style="font-size: 14px;">The module evaluates the weights and bias it receives; it does not initialize, train, or update them. Linear and nonlinear mode must therefore begin from the same supplied parameter values.</span>

<span style="font-size: 14px;">This separation keeps the forward calculation focused. Parameter learning belongs to a later optimization step, while this module only maps its current inputs and parameters to one scalar output.</span>

---

## <span style="font-size: 16px;">Shared dtype and device</span>

<span style="font-size: 14px;">Inputs, weights, and bias share one floating-point dtype and device. The returned scalar tensor preserves both; no dtype promotion is required.</span>

<span style="font-size: 14px;">All arithmetic remains on the shared device, and the returned value is a scalar tensor rather than a Python number. The supplied tensors remain unchanged because the arithmetic produces new values.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Applying tanh in both modes.** Linear mode returns the preactivation without an activation.</span>
* <span style="font-size: 14px;">**Adding the bias more than once.** The scalar bias shifts the summed weighted contributions exactly once.</span>
* <span style="font-size: 14px;">**Misaligning inputs and weights.** Each input must multiply the weight at the same position.</span>
* <span style="font-size: 14px;">**Rejecting empty vectors.** Empty aligned tensors are valid and leave the bias as the complete preactivation.</span>
* <span style="font-size: 14px;">**Changing dtype.** The output must preserve the shared input dtype.</span>
* <span style="font-size: 14px;">**Converting the result to a Python scalar.** The contract requires a scalar tensor on the shared device.</span>

<span style="font-size: 14px;">The module has one stable core, the weighted sum plus bias, and one controlled choice at the output: preserve the linear response or pass it through tanh.</span>

---
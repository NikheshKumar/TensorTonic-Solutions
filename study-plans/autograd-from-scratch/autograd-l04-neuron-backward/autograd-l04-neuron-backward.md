# <span style="font-size: 20px;">Backpropagate Through a Scalar Neuron</span>

<span style="font-size: 14px;">A neuron combines several inputs into one preactivation, passes that value through tanh, and produces a scalar output. Backpropagation answers the reverse question: when the final loss changes with the neuron output, how much responsibility belongs to each input, each weight, and the bias?</span>

---

## <span style="font-size: 16px;">The forward calculation</span>

<span style="font-size: 14px;">The neuron first forms a weighted sum and adds the bias:</span>

$$
a = \sum_{i=1}^{n} w_i x_i + b
$$

<span style="font-size: 14px;">It then applies tanh:</span>

$$
y = \tanh(a)
$$

<span style="font-size: 14px;">The preactivation $a$ collects all linear contributions. The output $y$ is the bounded value passed onward through the graph. The backward calculation uses this forward relationship in reverse.</span>

---

## <span style="font-size: 16px;">The gradient arriving from later computation</span>

<span style="font-size: 14px;">The neuron is usually not the final loss. Later operations produce an upstream gradient that measures how sensitive the loss is to the neuron output:</span>

$$
g = \frac{\partial L}{\partial y}
$$

<span style="font-size: 14px;">This value may be positive, negative, or zero. It tells the neuron how strongly the rest of the graph depends on $y$. The neuron combines that upstream influence with its own local derivative.</span>

---

## <span style="font-size: 16px;">From the output back to the preactivation</span>

<span style="font-size: 14px;">The local derivative of tanh can be written using the saved output:</span>

$$
\frac{\partial y}{\partial a} = 1-y^2
$$

<span style="font-size: 14px;">Applying the chain rule gives the gradient with respect to the preactivation:</span>

$$
\delta
=
\frac{\partial L}{\partial a}
=
\frac{\partial L}{\partial y}
\frac{\partial y}{\partial a}
=
g(1-y^2)
$$

<span style="font-size: 14px;">The single scalar $delta$ is the central backward signal for this neuron. Once it is known, every input and parameter gradient follows from the weighted sum.</span>

---

## <span style="font-size: 16px;">Gradients for inputs, weights, and bias</span>

<span style="font-size: 14px;">Inside the preactivation, input $x_i$ is multiplied by weight $w_i$. Holding everything else fixed gives:</span>

$$
\frac{\partial a}{\partial x_i}=w_i,
\qquad
\frac{\partial a}{\partial w_i}=x_i,
\qquad
\frac{\partial a}{\partial b}=1
$$

<span style="font-size: 14px;">Multiplying each local derivative by $delta$ produces the requested gradients:</span>

$$
\frac{\partial L}{\partial x_i}=\delta w_i
$$

$$
\frac{\partial L}{\partial w_i}=\delta x_i
$$

$$
\frac{\partial L}{\partial b}=\delta
$$

<span style="font-size: 14px;">The input-gradient vector aligns with the weights because each input receives the gradient through its multiplying weight. The weight-gradient vector aligns with the inputs because each weight receives the gradient through its multiplying input.</span>

---

## <span style="font-size: 16px;">A complete numerical example</span>

<span style="font-size: 14px;">Consider two inputs. The first has value $2$ and weight $-3$, while the second has value $0$ and weight $1$. Choose the bias so that the preactivation is approximately $0.881374$. The tanh output is then approximately:</span>

$$
y \approx 0.707107
$$

<span style="font-size: 14px;">With an upstream gradient of $1$, the preactivation gradient is:</span>

$$
\delta = 1\left(1-(0.707107)^2\right) \approx 0.5
$$

<span style="font-size: 14px;">The two input gradients are:</span>

$$
\frac{\partial L}{\partial x_1}=0.5(-3)=-1.5,
\qquad
\frac{\partial L}{\partial x_2}=0.5(1)=0.5
$$

<span style="font-size: 14px;">The two weight gradients are:</span>

$$
\frac{\partial L}{\partial w_1}=0.5(2)=1,
\qquad
\frac{\partial L}{\partial w_2}=0.5(0)=0
$$

<span style="font-size: 14px;">The bias gradient is $0.5$. A zero input therefore produces a zero gradient for its weight, although the corresponding input gradient can still be nonzero because that path also depends on the weight.</span>

---

## <span style="font-size: 16px;">Empty vectors and dtype promotion</span>

<span style="font-size: 14px;">When the input and weight vectors are empty, the bias is the entire preactivation. There are no input or weight gradients to return, but the bias still receives the tanh-scaled upstream gradient.</span>

<span style="font-size: 14px;">The supplied tensors may use different floating dtypes. Arithmetic must use their promoted input dtype while preserving the shared device. This keeps the output and every gradient consistent with PyTorch's type-promotion rules rather than silently forcing one original dtype onto all values.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Stopping at the tanh derivative.** The factor $1-y^2$ must also be multiplied by the upstream gradient.</span>
* <span style="font-size: 14px;">**Swapping the two vector rules.** Input gradients multiply $delta$ by weights, while weight gradients multiply $delta$ by inputs.</span>
* <span style="font-size: 14px;">**Forgetting the bias path.** Since the bias enters the preactivation with coefficient one, its gradient is exactly $delta$.</span>
* <span style="font-size: 14px;">**Assuming nonempty vectors.** An empty neuron still has a valid output and bias gradient.</span>
* <span style="font-size: 14px;">**Ignoring dtype promotion.** Mixed floating dtypes must produce arithmetic in the promoted dtype on the original device.</span>

<span style="font-size: 14px;">Backpropagation through the neuron is controlled by one shared scalar. The upstream gradient passes through tanh to form $delta$, and that value fans out through the weighted sum according to each local derivative.</span>

---
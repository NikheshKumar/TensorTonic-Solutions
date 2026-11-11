# <span style="font-size: 20px;">Evaluate a Dense Layer</span>

<span style="font-size: 14px;">A dense layer evaluates several neurons that all receive the same input vector. Each neuron has its own row of weights and its own bias, so the layer produces one output value per neuron.</span>

---

## <span style="font-size: 16px;">From one neuron to a layer</span>

<span style="font-size: 14px;">For neuron $j$, the preactivation is:</span>

$$
a_j=b_j+\sum_{i=1}^{n}W_{j,i}x_i
$$

<span style="font-size: 14px;">The input coordinate is indexed by $i$. The neuron is indexed by $j$. Every neuron reads the same input vector but uses a different weight row and bias entry.</span>

<span style="font-size: 14px;">Collecting all neuron equations gives the matrix form:</span>

$$
\mathbf{a}=W\mathbf{x}+\mathbf{b}
$$

---

## <span style="font-size: 16px;">How matrix rows map to neurons</span>

<span style="font-size: 14px;">The weight matrix has one row per neuron and one column per input. Row $j$ contains all weights used by neuron $j$. Bias entry $j$ belongs to that same neuron.</span>

<span style="font-size: 14px;">If the layer has $m$ neurons and the input width is $n$, then:</span>

$$
W\in\mathbb{R}^{m\times n}
$$

$$
\mathbf{x}\in\mathbb{R}^{n}
$$

$$
\mathbf{b}\in\mathbb{R}^{m}
$$

<span style="font-size: 14px;">The matrix-vector product therefore produces $m$ preactivations in neuron order.</span>

---

## <span style="font-size: 16px;">Linear and nonlinear layer modes</span>

<span style="font-size: 14px;">Linear mode returns the preactivation vector:</span>

$$
\mathbf{o}=\mathbf{a}
$$

<span style="font-size: 14px;">Nonlinear mode applies tanh independently to every coordinate:</span>

$$
o_j=\tanh(a_j)
$$

<span style="font-size: 14px;">Elementwise activation does not mix neurons. Each output changes only according to its own preactivation after the shared matrix-vector calculation is complete.</span>

---

## <span style="font-size: 16px;">A rectangular linear example</span>

<span style="font-size: 14px;">Consider three input values, $1$, $2$, and $-1$, feeding two neurons. The first neuron uses weights $1$, $0$, and $1$ with bias $0.5$. The second uses weights $0$, $2$, and $0$ with bias $-0.5$.</span>

<span style="font-size: 14px;">The first preactivation is:</span>

$$
a_1=(1)(1)+(0)(2)+(1)(-1)+0.5=0.5
$$

<span style="font-size: 14px;">The second preactivation is:</span>

$$
a_2=(0)(1)+(2)(2)+(0)(-1)-0.5=3.5
$$

<span style="font-size: 14px;">The linear output contains $0.5$ followed by $3.5$, preserving the original row and bias order.</span>

---

## <span style="font-size: 16px;">A single nonlinear neuron</span>

<span style="font-size: 14px;">A dense layer may contain only one neuron. For inputs $2$ and $1$, weights $1$ and $-1$, and zero bias:</span>

$$
a=(1)(2)+(-1)(1)=1
$$

<span style="font-size: 14px;">Nonlinear mode returns:</span>

$$
\tanh(1)\approx 0.761594
$$

<span style="font-size: 14px;">The result remains a one-dimensional tensor with one element, not a zero-dimensional scalar, because the layer still has one declared neuron.</span>

---

## <span style="font-size: 16px;">Zero weights and bias-only outputs</span>

<span style="font-size: 14px;">If a neuron's weight row contains only zeros, none of the inputs contribute to its preactivation. Its output in linear mode is exactly its bias.</span>

$$
W_{j,i}=0\ \text{for every }i
\quad\Longrightarrow\quad
a_j=b_j
$$

<span style="font-size: 14px;">Different zero-weight rows can still produce different outputs because each neuron has its own aligned bias.</span>

---

## <span style="font-size: 16px;">Neuron order and independence</span>

<span style="font-size: 14px;">Every output coordinate belongs to one matrix row and one bias entry. Changing the parameters of neuron $j$ changes output $j$ without changing how the other rows form their preactivations.</span>

<span style="font-size: 14px;">The neurons share the input vector, not their parameters. Preserving row order in the output is therefore part of the layer's meaning, not merely a presentation choice.</span>

---

## <span style="font-size: 16px;">Shared dtype, device, and immutability</span>

<span style="font-size: 14px;">The input vector, weight matrix, and bias vector share one floating-point dtype and device. The output preserves both.</span>

<span style="font-size: 14px;">Matrix multiplication, bias addition, and optional tanh remain on the shared device. These operations produce new tensors, so the supplied input, weight, and bias tensors are not modified.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Treating columns as neurons.** Each matrix row, not each column, supplies one neuron's weights.</span>
* <span style="font-size: 14px;">**Misaligning biases.** Bias entry $j$ must be added to the result from weight row $j$.</span>
* <span style="font-size: 14px;">**Applying tanh before adding bias.** The activation receives the complete preactivation $W\mathbf{x}+\mathbf{b}$.</span>
* <span style="font-size: 14px;">**Applying tanh in linear mode.** The mode flag controls whether the preactivation vector is transformed.</span>
* <span style="font-size: 14px;">**Collapsing a one-neuron result to a scalar.** The layer output remains a vector in neuron order.</span>
* <span style="font-size: 14px;">**Changing dtype.** The result must preserve the shared input dtype.</span>

<span style="font-size: 14px;">A dense layer is a collection of scalar neurons expressed compactly as one matrix-vector product, one aligned bias addition, and an optional elementwise activation.</span>

---
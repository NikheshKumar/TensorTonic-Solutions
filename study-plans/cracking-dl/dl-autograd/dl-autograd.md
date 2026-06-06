# <span style="font-size: 20px;">Computational Graph & Autograd</span>

<span style="font-size: 14px;">Automatic differentiation (autograd) is the engine behind every modern deep learning framework. Rather than deriving gradients by hand (as in Problem 5) or approximating them numerically, autograd records the computation as a directed acyclic graph (DAG) and applies the chain rule algorithmically. This is how PyTorch, JAX, and TensorFlow compute gradients for arbitrary computations.</span>

---

## <span style="font-size: 16px;">Computational Graphs</span>

<span style="font-size: 14px;">A computational graph represents a mathematical expression as a DAG where:</span>

* <span style="font-size: 14px;">**Leaf nodes** are inputs (variables or constants)</span>
* <span style="font-size: 14px;">**Internal nodes** are operations (add, mul, relu, ...)</span>
* <span style="font-size: 14px;">**Edges** encode data dependencies</span>

<span style="font-size: 14px;">For</span> $f(x, y) = (x \cdot y) + x$<span style="font-size: 14px;">:</span>

<span style="font-size: 14px;">1. Node A = x (input)</span>
<span style="font-size: 14px;">2. Node B = y (input)</span>
<span style="font-size: 14px;">3. Node C = A * B (mul)</span>
<span style="font-size: 14px;">4. Node D = C + A (add)</span>

<span style="font-size: 14px;">Notice that node A has two consumers (the mul and the add). During backpropagation, its gradient is the sum of contributions from both paths.</span>

---

## <span style="font-size: 16px;">Forward Mode vs Reverse Mode</span>

<span style="font-size: 14px;">There are two ways to propagate derivatives through a graph:</span>

<span style="font-size: 14px;">**Forward mode**: computes</span> $\partial \text{output} / \partial x_i$ <span style="font-size: 14px;">by propagating derivatives forward from a single input. Cost: one forward pass per input variable. Efficient when outputs >> inputs.</span>

<span style="font-size: 14px;">**Reverse mode**: computes</span> $\partial \text{output} / \partial x_i$ <span style="font-size: 14px;">for ALL inputs in a single backward pass. Cost: one backward pass total. Efficient when inputs >> outputs.</span>

<span style="font-size: 14px;">Neural networks have millions of parameters (inputs to the gradient) and a single scalar loss (output). Reverse mode is therefore exponentially more efficient - this is why all DL frameworks use reverse-mode autodiff (backpropagation is a special case).</span>

---

## <span style="font-size: 16px;">Topological Sort</span>

<span style="font-size: 14px;">The backward pass must visit nodes in reverse topological order: a node's gradient must be fully accumulated before it propagates to its children. This ensures that when we call a node's backward function, its</span> `.grad` <span style="font-size: 14px;">field contains the complete gradient from all downstream consumers.</span>

<span style="font-size: 14px;">The standard implementation uses a post-order DFS from the output node to build the topological ordering, then iterates in reverse:</span>

<span style="font-size: 14px;">1. Start DFS at output node</span>
<span style="font-size: 14px;">2. Recursively visit all children first</span>
<span style="font-size: 14px;">3. Append current node after children are processed</span>
<span style="font-size: 14px;">4. Iterate the resulting list in reverse for the backward pass</span>

---

## <span style="font-size: 16px;">Gradient Accumulation</span>

<span style="font-size: 14px;">The most subtle part of autograd. When a variable is used in multiple places:</span>

$$
f(x) = x \cdot x + x
$$

<span style="font-size: 14px;">There are three paths from the output back to</span> $x$<span style="font-size: 14px;">: two through the multiplication (left and right arguments) and one through the addition. The total gradient is:</span>

$$
\frac{df}{dx} = \underbrace{x}_{\text{right of mul}} + \underbrace{x}_{\text{left of mul}} + \underbrace{1}_{\text{add}} = 2x + 1
$$

<span style="font-size: 14px;">In code, this means initializing every node's gradient to 0 and using</span> `+=` <span style="font-size: 14px;">(not</span> `=`<span style="font-size: 14px;">) in every backward function. The</span> `+=` <span style="font-size: 14px;">is what makes gradient accumulation work.</span>

---

## <span style="font-size: 16px;">Local Gradients for Each Operation</span>

<span style="font-size: 14px;">Each operation only needs to know its own local gradient rule. The chain rule handles the rest:</span>

| Operation | Forward | Backward (given upstream gradient $g$) |
|---|---|---|
| $c = a + b$ | $c = a + b$ | $\bar{a}$ += $g$, $\bar{b}$ += $g$ |
| $c = a \times b$ | $c = a \times b$ | $\bar{a}$ += $b \cdot g$, $\bar{b}$ += $a \cdot g$ |
| $c = a^n$ | $c = a^n$ | $\bar{a}$ += $n \cdot a^{n-1} \cdot g$ |
| $c = \text{relu}(a)$ | $c = \max(0, a)$ | $\bar{a}$ += $\mathbb{1}[a > 0] \cdot g$ |
| $c = -a$ | $c = -a$ | $\bar{a}$ += $-g$ |

<span style="font-size: 14px;">Here</span> $\bar{a}$ <span style="font-size: 14px;">denotes the gradient accumulated at node</span> $a$<span style="font-size: 14px;">.</span>

---

## <span style="font-size: 16px;">Connection to PyTorch</span>

<span style="font-size: 14px;">PyTorch's autograd is exactly this algorithm at scale:</span>

* <span style="font-size: 14px;">Every</span> `torch.Tensor` <span style="font-size: 14px;">with</span> `requires_grad=True` <span style="font-size: 14px;">is a node in the graph</span>
* <span style="font-size: 14px;">Operations like</span> `+, *, @` <span style="font-size: 14px;">create new nodes with</span> `grad_fn` <span style="font-size: 14px;">attributes that store the backward function</span>
* <span style="font-size: 14px;">Calling</span> `loss.backward()` <span style="font-size: 14px;">triggers the reverse topological traversal</span>
* <span style="font-size: 14px;">Gradients accumulate in</span> `.grad` <span style="font-size: 14px;">attributes (this is why you must call</span> `optimizer.zero_grad()` <span style="font-size: 14px;">before each backward pass)</span>

<span style="font-size: 14px;">JAX takes a different approach (functional transforms) but the underlying math is identical.</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

<span style="font-size: 14px;">Common follow-up questions in deep learning interviews:</span>


* <span style="font-size: 14px;">**Why must gradients accumulate (+=) instead of overwrite (=)?** Because a node can feed into multiple downstream operations. Each downstream operation contributes to the total derivative via the chain rule. Overwriting would lose earlier contributions. This is also why PyTorch requires</span> `zero_grad()` <span style="font-size: 14px;">- without it, gradients from the previous iteration accumulate</span>
* <span style="font-size: 14px;">**What is the relationship between backpropagation and reverse-mode autodiff?** Backpropagation IS reverse-mode autodiff applied to the specific computational graph of a neural network. The terms are often used interchangeably, but autodiff is more general (it works for any differentiable program, not just neural networks)</span>
* <span style="font-size: 14px;">**How does autograd handle in-place operations?** In-place ops (like</span> `x += 1`<span style="font-size: 14px;">) modify the data of an existing node, which can corrupt the saved values needed for backward. PyTorch detects this and raises an error. This is why in-place operations on tensors with gradients are discouraged</span>
* <span style="font-size: 14px;">**What is the computational cost of backward vs forward?** The backward pass visits every node once and performs a constant amount of work per node (the local gradient rule). Its cost is proportional to the forward pass - typically 2-3x the forward pass cost. This is independent of the number of parameters, which is what makes it practical for large models</span>
* <span style="font-size: 14px;">**Can autograd compute second derivatives?** Yes - by building a computational graph of the backward pass itself and differentiating through it. PyTorch supports this with</span> `create_graph=True` <span style="font-size: 14px;">in</span> `backward()`<span style="font-size: 14px;">. This is needed for some optimization methods (Hessian-vector products) and regularization techniques (gradient penalty in WGANs)</span>

---
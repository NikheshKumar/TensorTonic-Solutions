# <span style="font-size: 20px;">Create an Addition Value Node</span>

<span style="font-size: 14px;">An automatic differentiation system needs more than the final numeric answer. It must also remember how that answer was produced so that gradients can later travel back to the inputs. A value node stores both pieces of information: the scalar result and its connection to the earlier nodes.</span>

<span style="font-size: 14px;">This problem constructs one node for a single addition:</span>

$$
z=x+y
$$

<span style="font-size: 14px;">The supplied left and right records are leaf nodes. They already hold scalar data, but they do not describe an operation that created them. The new output is an operation node because it was created by adding those two leaves.</span>

---

## <span style="font-size: 16px;">What the output node must remember</span>

<span style="font-size: 14px;">The returned record has exactly five fields, and each has a distinct job:</span>

- <span style="font-size: 14px;">**ID** identifies the new node and must use the supplied output ID.</span>
- <span style="font-size: 14px;">**data** stores the forward value, which is the sum of the two leaf values.</span>
- <span style="font-size: 14px;">**gradient** stores the gradient that will eventually reach this node.</span>
- <span style="font-size: 14px;">**operation** records that addition created the node, so its value is the plus symbol.</span>
- <span style="font-size: 14px;">**parents** stores the original left and right leaf objects in that order.</span>

<span style="font-size: 14px;">Together, **operation** and **parents** are a small description of the graph edge that produced the output. Without them, the program would know that the result is five, for example, but it would not know whether five came from addition or which earlier values participated.</span>

---

## <span style="font-size: 16px;">The forward value</span>

<span style="font-size: 14px;">The numeric work is intentionally simple. Read the left leaf data and the right leaf data as 64-bit floating-point values, add them, and store the result as a Python float.</span>

<span style="font-size: 14px;">For these leaves:</span>

* <span style="font-size: 14px;">**Left leaf:** identifier $a$ with scalar value $2$</span>
* <span style="font-size: 14px;">**Right leaf:** identifier $b$ with scalar value $3$</span>

<span style="font-size: 14px;">the output data is:</span>

$$
z=2+3=5
$$

<span style="font-size: 14px;">If the supplied output identifier is $c$, the new node records that node $c$ contains five and was created by adding nodes $a$ and $b$.</span>

---

## <span style="font-size: 16px;">Why the gradient starts at zero</span>

<span style="font-size: 14px;">Creating the node performs only the forward calculation. No backward pass has happened yet, so no gradient has been propagated into the result. Its **gradient** field therefore begins at $0$.</span>

<span style="font-size: 14px;">Later, when a backward calculation reaches this addition, the local derivatives will be simple:</span>

$$
\frac{\partial z}{\partial x}=1,
\qquad
\frac{\partial z}{\partial y}=1
$$

<span style="font-size: 14px;">That fact explains why remembering both parents will matter, but this task stops before performing any backward propagation. It only prepares the node with a clean initial gradient and the necessary graph history.</span>

---

## <span style="font-size: 16px;">Object identity and parent order</span>

<span style="font-size: 14px;">The parents list must contain the exact objects passed to the function. Creating copied dictionaries with the same contents is not equivalent. A computation graph connects node objects, and later code may update or inspect those same objects while traversing the graph.</span>

<span style="font-size: 14px;">The left leaf must remain the first parent and the right leaf must remain the second. Addition gives the same numeric answer if the operands are swapped, but graph construction should still preserve the operation as it was written. Consistent operand order becomes especially important when the same representation is used for operations whose inputs are not interchangeable.</span>

---

## <span style="font-size: 16px;">Create a new record without mutation</span>

<span style="font-size: 14px;">The output node is a new object. The leaf records must remain exactly as they were before the call. Adding fields such as **gradient**, **operation**, or **parents** to a leaf would change the caller's data and blur the difference between the existing inputs and the newly created result.</span>

<span style="font-size: 14px;">A reliable construction sequence is:</span>

1. <span style="font-size: 14px;">Read the two scalar data values without editing either record.</span>
2. <span style="font-size: 14px;">Add them using 64-bit floating-point arithmetic.</span>
3. <span style="font-size: 14px;">Create a new record with exactly the five required field names.</span>
4. <span style="font-size: 14px;">Store $0$ as the initial gradient and $+$ as the operation.</span>
5. <span style="font-size: 14px;">Place the original left and right objects into the parents list in order.</span>

---

## <span style="font-size: 16px;">Common mistakes</span>

- <span style="font-size: 14px;">Storing only parent IDs loses the required references to the original leaf objects.</span>
- <span style="font-size: 14px;">Copying the leaves preserves their visible contents but breaks the required object identity.</span>
- <span style="font-size: 14px;">Adding extra fields violates the exact output-record contract.</span>
- <span style="font-size: 14px;">Using a word such as **add** instead of the plus symbol records the wrong operation label.</span>
- <span style="font-size: 14px;">Mutating a leaf to turn it into the output destroys the original graph inputs.</span>
- <span style="font-size: 14px;">Trying to propagate gradients now performs work that belongs to a later backward step.</span>

<span style="font-size: 14px;">This one addition node establishes the basic autograd pattern: compute a scalar value during the forward pass, create a fresh node for that value, and retain the exact graph connections needed for later differentiation.</span>

---
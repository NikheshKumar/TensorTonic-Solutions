# <span style="font-size: 20px;">Create a Multiplication Value Node</span>

<span style="font-size: 14px;">A scalar autograd engine must remember both the number produced by an operation and the earlier values that produced it. A multiplication node stores the product from the forward pass together with the graph connection needed for a later backward pass.</span>

$$
z = xy
$$

<span style="font-size: 14px;">The two supplied records are leaf nodes. The returned record is a new operation node whose value is their product.</span>

---

## <span style="font-size: 16px;">What the new node represents</span>

<span style="font-size: 14px;">The output record has exactly five fields. Each field carries one part of the node's state:</span>

* <span style="font-size: 14px;">**ID:** the supplied identifier for the newly created node</span>
* <span style="font-size: 14px;">**Data:** the product of the two input values</span>
* <span style="font-size: 14px;">**Gradient:** zero, because no backward pass has reached the node yet</span>
* <span style="font-size: 14px;">**Operation:** the multiplication symbol</span>
* <span style="font-size: 14px;">**Parents:** the original left and right leaf objects, stored in that order</span>

<span style="font-size: 14px;">The data field answers what was computed. The operation and parent fields answer how it was computed.</span>

---

## <span style="font-size: 16px;">The forward calculation</span>

<span style="font-size: 14px;">Read the scalar data from each leaf and multiply the two values using 64-bit floating-point arithmetic.</span>

<span style="font-size: 14px;">For a left leaf containing $2$ and a right leaf containing $3$, the new data value is:</span>

$$
z = 2 \times 3 = 6
$$

<span style="font-size: 14px;">A negative factor changes the sign of the product. A zero factor makes the product zero. These are ordinary multiplication rules, and the graph metadata does not change them.</span>

---

## <span style="font-size: 16px;">Why both parents must be retained</span>

<span style="font-size: 14px;">Multiplication has a different local derivative for each input:</span>

$$
\frac{\partial z}{\partial x} = y,
\qquad
\frac{\partial z}{\partial y} = x
$$

<span style="font-size: 14px;">A later backward pass therefore needs both original parent values. The gradient sent to the left parent depends on the right parent's data, while the gradient sent to the right parent depends on the left parent's data.</span>

<span style="font-size: 14px;">This task does not perform that backward calculation. It only preserves the exact objects that will make the calculation possible later.</span>

---

## <span style="font-size: 16px;">Identity and order are part of the graph</span>

<span style="font-size: 14px;">The parent list must contain the original leaf objects rather than copies with identical contents. Graph traversal works with the nodes themselves, so a copied record would create a different object and break the required identity relationship.</span>

<span style="font-size: 14px;">The left leaf remains first and the right leaf remains second. Multiplication is numerically commutative, but preserving operand order gives every operation node a consistent representation. The same convention also works for later operations where swapping operands changes the result.</span>

---

## <span style="font-size: 16px;">A readable example</span>

* <span style="font-size: 14px;">The left leaf has identifier $a$ and scalar value $2$.</span>
* <span style="font-size: 14px;">The right leaf has identifier $b$ and scalar value $3$.</span>
* <span style="font-size: 14px;">The new node uses identifier $c$ and stores the product $6$.</span>
* <span style="font-size: 14px;">Its gradient starts at zero, its operation is multiplication, and its parents are the original $a$ and $b$ leaf objects in that order.</span>

<span style="font-size: 14px;">The two leaf records remain unchanged. Only the new output record contains the multiplication history.</span>

---

## <span style="font-size: 16px;">Implementation order</span>

* <span style="font-size: 14px;">Read the two scalar data values without modifying either leaf.</span>
* <span style="font-size: 14px;">Convert the values to the required floating-point precision and multiply them.</span>
* <span style="font-size: 14px;">Create a fresh record with exactly the five required fields.</span>
* <span style="font-size: 14px;">Initialize the gradient to zero and record multiplication as the operation.</span>
* <span style="font-size: 14px;">Store the original left and right objects as ordered parents.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Copying the parent records.** Equal contents do not satisfy the requirement that the exact input objects remain connected to the output.</span>
* <span style="font-size: 14px;">**Storing only parent identifiers.** This problem requires the parent objects themselves, unlike later graph representations that may store identifiers.</span>
* <span style="font-size: 14px;">**Mutating a leaf.** Turning an input record into the output destroys the original graph structure.</span>
* <span style="font-size: 14px;">**Computing gradients immediately.** The node starts with a zero gradient because backward propagation belongs to a later step.</span>

<span style="font-size: 14px;">The multiplication node is small, but it contains the complete forward-pass pattern used throughout scalar autograd: compute a value, create a fresh node, and retain the precise history required for differentiation.</span>

---
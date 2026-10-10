# <span style="font-size: 20px;">SVM with Hinge Loss (SGD)</span>

<span style="font-size: 14px;">Support Vector Machines are among the most well-studied classifiers in machine learning. The original SVM formulation uses a quadratic program to find the maximum-margin hyperplane, but the same solution can be approximated efficiently using stochastic subgradient descent on the hinge loss. This is the practical approach used in large-scale SVM implementations like PEGASOS and liblinear.</span>

---

## <span style="font-size: 16px;">Maximum Margin Principle</span>

- <span style="font-size: 14px;">Among all hyperplanes that separate the data, the SVM finds the one with the largest margin</span>
- <span style="font-size: 14px;">The margin is the distance from the hyperplane to the nearest data point on either side</span>
- <span style="font-size: 14px;">For weight vector</span> $w$ <span style="font-size: 14px;">and bias</span> $b$<span style="font-size: 14px;">, the geometric margin of point</span> $i$ <span style="font-size: 14px;">is</span> $y_i(w \cdot x_i + b) / \|w\|$
- <span style="font-size: 14px;">Maximizing the margin is equivalent to minimizing</span> $\|w\|^2$ <span style="font-size: 14px;">subject to all points being correctly classified with margin at least 1</span>
- <span style="font-size: 14px;">Larger margins lead to better generalization (PAC learning theory)</span>

---

## <span style="font-size: 16px;">Hinge Loss</span>

<span style="font-size: 14px;">The hinge loss relaxes the hard constraint to allow some misclassifications:</span>

$$
\ell_{\text{hinge}}(w, b; x_i, y_i) = \max(0, 1 - y_i(w \cdot x_i + b))
$$

- <span style="font-size: 14px;">Zero when the point is on the correct side with margin</span> $\geq 1$ <span style="font-size: 14px;">(well classified)</span>
- <span style="font-size: 14px;">Linear penalty when the margin is less than 1 (within the margin band or misclassified)</span>
- <span style="font-size: 14px;">Not differentiable at</span> $y_i(w \cdot x_i + b) = 1$<span style="font-size: 14px;">, which is why we use subgradients</span>

---

## <span style="font-size: 16px;">Regularized Objective</span>

<span style="font-size: 14px;">The SVM training objective combines hinge loss with L2 regularization:</span>

$$
L(w, b) = \frac{\lambda}{2} \|w\|^2 + \frac{1}{n} \sum_{i=1}^{n} \max(0, 1 - y_i(w \cdot x_i + b))
$$

- <span style="font-size: 14px;">The regularization term</span> $\frac{\lambda}{2} \|w\|^2$ <span style="font-size: 14px;">penalizes large weights, which corresponds to maximizing the margin</span>
- <span style="font-size: 14px;">Small</span> $\lambda$ <span style="font-size: 14px;">allows a narrower margin (fits data more tightly)</span>
- <span style="font-size: 14px;">Large</span> $\lambda$ <span style="font-size: 14px;">forces a wider margin at the cost of more hinge loss</span>
- <span style="font-size: 14px;">This is the "soft margin" SVM, equivalent to the dual formulation with slack variables</span>

---

## <span style="font-size: 16px;">Subgradient Descent</span>

<span style="font-size: 14px;">Since the hinge loss is not differentiable everywhere, we use subgradients. For a single point</span> $(x_i, y_i)$<span style="font-size: 14px;">:</span>

$$
\begin{aligned}
&\text{If } y_i(w \cdot x_i + b) < 1\text{:} \\
&\quad \nabla_w = \lambda w - y_i x_i,\quad \nabla_b = -y_i
\end{aligned}
$$

$$
\text{If } y_i(w \cdot x_i + b) \geq 1: \quad \nabla_w = \lambda w, \quad \nabla_b = 0
$$

<span style="font-size: 14px;">The update rule applies the learning rate to the subgradient:</span>

$$
w \leftarrow w - \alpha \nabla_w, \quad b \leftarrow b - \alpha \nabla_b
$$

---

## <span style="font-size: 16px;">Support Vectors</span>

- <span style="font-size: 14px;">Points with margin exactly equal to 1 lie on the margin boundary and are called **support vectors**</span>
- <span style="font-size: 14px;">Points with margin less than 1 are either inside the margin band or misclassified</span>
- <span style="font-size: 14px;">Points with margin greater than 1 are well outside the margin and do not influence the solution</span>
- <span style="font-size: 14px;">The decision boundary depends only on the support vectors, making SVM robust to outliers far from the boundary</span>

---

## <span style="font-size: 16px;">Comparison with Other Linear Classifiers</span>

- <span style="font-size: 14px;">**Perceptron**: uses a step function loss (0 if correct, 1 if wrong). No margin concept, no regularization</span>
- <span style="font-size: 14px;">**Logistic regression**: uses log-loss, which is smooth and yields probability estimates. Never fully "satisfied" even with correct classifications</span>
- <span style="font-size: 14px;">**SVM**: uses hinge loss, which is zero for well-classified points. Focuses only on points near or on the wrong side of the margin</span>

---

## <span style="font-size: 16px;">Kernel Extension</span>

- <span style="font-size: 14px;">Linear SVM can only learn linear boundaries</span>
- <span style="font-size: 14px;">The kernel trick maps data to a higher-dimensional space where it becomes linearly separable</span>
- <span style="font-size: 14px;">Common kernels: RBF (Gaussian), polynomial, sigmoid</span>
- <span style="font-size: 14px;">Kernelized SVM uses the dual formulation and is typically solved with SMO, not SGD</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

- <span style="font-size: 14px;">**Q: Why hinge loss instead of 0/1 loss?**</span>
  <span style="font-size: 14px;">A: The 0/1 loss is non-convex and non-differentiable, making optimization intractable. Hinge loss is a convex upper bound on the 0/1 loss, so minimizing it approximately minimizes classification error while being tractable to optimize</span>

- <span style="font-size: 14px;">**Q: What is the relationship between C and lambda?**</span>
  <span style="font-size: 14px;">A: In the standard SVM formulation, $C$ controls the penalty for slack variables: $\frac{1}{2}\|w\|^2 + C \sum \xi_i$. The regularized form uses $\frac{\lambda}{2}\|w\|^2 + \frac{1}{n}\sum \ell_i$. The relationship is $\lambda = 1/(nC)$</span>

- <span style="font-size: 14px;">**Q: How does the SVM handle non-separable data?**</span>
  <span style="font-size: 14px;">A: Soft-margin SVM allows some points to violate the margin (slack variables). The regularization parameter controls the tradeoff between margin width and violations. All practical SVMs are soft-margin</span>

- <span style="font-size: 14px;">**Q: Why use SGD instead of the dual?**</span>
  <span style="font-size: 14px;">A: SGD scales to millions of examples with $O(d)$ per-update cost. The dual (quadratic program) has $O(n^2)$ to $O(n^3)$ complexity. For large-scale linear SVM, SGD (PEGASOS) or coordinate descent (liblinear) are standard</span>

- <span style="font-size: 14px;">**Q: SVM vs logistic regression?**</span>
  <span style="font-size: 14px;">A: SVM uses hinge loss (sparse solution, many zero-loss points), logistic regression uses log-loss (all points contribute). SVM typically works better with small datasets or high dimensions; logistic regression gives calibrated probabilities. In practice, both achieve similar accuracy with proper tuning</span>

---
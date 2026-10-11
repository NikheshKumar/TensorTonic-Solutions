## The Least Squares Problem

<span style="font-size: 14px;">Given an overdetermined system $Ax = b$ where $A \in \mathbb{R}^{m \times n}$ with $m > n$ (more equations than unknowns), there is generally no exact solution. Instead, we seek the $x$ that minimizes the squared residual:</span>

$$
x^* = \arg\min_x \|Ax - b\|^2 = \arg\min_x \sum_{i=1}^m (a_i^T x - b_i)^2
$$

---

## Derivation of Normal Equations

<span style="font-size: 14px;">Expanding the objective: $f(x) = \|Ax - b\|^2 = (Ax-b)^T(Ax-b) = x^TA^TAx - 2b^TAx + b^Tb$. Taking the gradient and setting it to zero:</span>

$$
\nabla_x f = 2A^TAx - 2A^Tb = 0 \implies A^T A x = A^T b
$$

<span style="font-size: 14px;">When $A$ has full column rank, $A^T A$ is invertible and the unique solution is:</span>

$$
x^* = (A^T A)^{-1} A^T b
$$

<span style="font-size: 14px;">The second derivative (Hessian) is $2A^TA$, which is positive semidefinite, confirming this is a minimum.</span>

---

## Geometric Interpretation

<span style="font-size: 14px;">The least squares solution $Ax^*$ is the orthogonal projection of $b$ onto the column space of $A$. The residual $r = b - Ax^*$ is orthogonal to $\mathcal{C}(A)$, which is exactly the condition $A^T r = 0$ (the normal equations). This is the fundamental theorem: among all vectors in $\mathcal{C}(A)$, the projection $Ax^*$ is closest to $b$. The projection matrix is $P = A(A^TA)^{-1}A^T$, and the fitted values are $\hat{b} = Pb$.</span>

---

## When $A^TA$ Is Singular

<span style="font-size: 14px;">If $A$ does not have full column rank, $A^TA$ is singular and $(A^TA)^{-1}$ does not exist. In this case, there are infinitely many minimizers of $\|Ax - b\|^2$. The pseudoinverse $A^+b$ selects the unique minimum-norm solution among them. In practice, this situation arises with multicollinear features and is handled by regularization or the SVD-based pseudoinverse.</span>

---

## QR-Based Solution

<span style="font-size: 14px;">Solving via the normal equations can be numerically unstable because forming $A^T A$ squares the condition number: $\kappa(A^T A) = \kappa(A)^2$. A more stable approach uses QR decomposition:</span>

$$
A = QR \implies Rx = Q^T b
$$

<span style="font-size: 14px;">Since $R$ is upper triangular, this system is easily solved by back-substitution. The condition number of $R$ equals $\kappa(A)$, not $\kappa(A)^2$. This is why `np.linalg.lstsq` uses SVD/QR internally rather than normal equations.</span>

---

## Polynomial Fitting as Least Squares

<span style="font-size: 14px;">Fitting a degree-$d$ polynomial $p(t) = c_0 + c_1 t + \cdots + c_d t^d$ to data points $(t_1, y_1), \ldots, (t_m, y_m)$ is a least squares problem with the Vandermonde matrix:</span>

$$
A = \begin{pmatrix} 1 & t_1 & t_1^2 & \cdots & t_1^d \\ \vdots & \vdots & \vdots & & \vdots \\ 1 & t_m & t_m^2 & \cdots & t_m^d \end{pmatrix}, \qquad Ac = y
$$

<span style="font-size: 14px;">The least squares solution $c^* = (A^TA)^{-1}A^Ty$ gives the polynomial coefficients that minimize total squared error.</span>

---

## Regularized Least Squares (Ridge Regression)

<span style="font-size: 14px;">Adding an $L_2$ penalty modifies the objective to $\|Ax - b\|^2 + \lambda\|x\|^2$, yielding the regularized normal equations:</span>

$$
(A^TA + \lambda I)x = A^Tb
$$

<span style="font-size: 14px;">The matrix $A^TA + \lambda I$ is always invertible for $\lambda > 0$, even when $A$ is rank-deficient. This trades a small increase in bias for a large reduction in variance, preventing overfitting. The effective condition number drops to $\kappa \approx (\sigma_1^2 + \lambda)/(\sigma_n^2 + \lambda)$, which is much better when $\sigma_n$ is small.</span>

---

## Applications in ML

* <span style="font-size: 14px;">**Linear regression**: the closed-form OLS estimator is exactly the least squares solution. Adding a column of ones to $X$ gives an intercept term.</span>
* <span style="font-size: 14px;">**Polynomial fitting**: fitting a degree-$d$ polynomial to $n$ data points is least squares with a Vandermonde matrix, as shown above</span>
* <span style="font-size: 14px;">**Regularization**: ridge regression modifies the normal equations to $(A^TA + \lambda I)x = A^Tb$, preventing overfitting and ensuring a unique solution</span>
* <span style="font-size: 14px;">**Signal processing**: fitting a linear combination of basis functions (Fourier, wavelets) to a signal is a least squares problem. The normal equations produce the best approximation in the $L_2$ sense.</span>
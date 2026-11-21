## Projection onto a Subspace

<span style="font-size: 14px;">Given a subspace $\mathcal{C}(A)$ (the column space of $A$), the orthogonal projection of a vector $b$ onto this subspace is the point $\hat{b} \in \mathcal{C}(A)$ that is closest to $b$. The error $b - \hat{b}$ is orthogonal to every vector in $\mathcal{C}(A)$.</span>

$$
\hat{b} = Pb, \qquad P = A(A^T A)^{-1} A^T
$$

---

## Derivation

<span style="font-size: 14px;">We want $\hat{b} = Ax$ for some $x$, and the residual $b - Ax$ must be orthogonal to $\mathcal{C}(A)$:</span>

$$
A^T(b - Ax) = 0 \implies A^T A x = A^T b \implies x = (A^T A)^{-1} A^T b
$$

<span style="font-size: 14px;">Therefore $\hat{b} = A(A^T A)^{-1} A^T b = Pb$. This derivation requires $A^T A$ to be invertible, which happens exactly when $A$ has full column rank.</span>

---

## Worked Example: Projection onto a Plane

<span style="font-size: 14px;">Let $A = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \end{pmatrix}$ (the $xy$-plane in $\mathbb{R}^3$) and $b = (3, 4, 5)^T$. Then $A^T A = I_2$, so $P = AA^T = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0 \end{pmatrix}$. The projection is $Pb = (3, 4, 0)^T$, which drops the $z$-component. The residual $(0, 0, 5)^T$ is orthogonal to the $xy$-plane, confirming correctness.</span>

---

## Properties of Projection Matrices

* <span style="font-size: 14px;">**Idempotent**: $P^2 = P$ - projecting twice gives the same result as projecting once. Proof: $P^2 = A(A^TA)^{-1}A^T A(A^TA)^{-1}A^T = A(A^TA)^{-1}A^T = P$.</span>
* <span style="font-size: 14px;">**Symmetric**: $P^T = P$ - the projection is self-adjoint</span>
* <span style="font-size: 14px;">**Eigenvalues**: all eigenvalues of $P$ are 0 or 1, with the number of 1s equal to $\mathrm{rank}(A)$</span>
* <span style="font-size: 14px;">**Trace and rank**: $\mathrm{tr}(P) = \mathrm{rank}(P) = \mathrm{rank}(A) = n$ (for full column rank $A$). Since eigenvalues are 0 or 1, the trace counts the dimension of the subspace projected onto.</span>
* <span style="font-size: 14px;">**Complementary projector**: $I - P$ projects onto the orthogonal complement $\mathcal{C}(A)^\perp$ (the left null space of $A$). It is also idempotent and symmetric.</span>

---

## Geometric Interpretation

<span style="font-size: 14px;">The projection $Pb$ decomposes any vector $b$ into two orthogonal components:</span>

$$
b = Pb + (I - P)b
$$

<span style="font-size: 14px;">where $Pb$ lies in $\mathcal{C}(A)$ and $(I-P)b$ lies in $\mathcal{C}(A)^\perp$. This is the fundamental decomposition behind least squares: the residual $b - A\hat{x}$ is orthogonal to the column space, making $A\hat{x}$ the best approximation to $b$. Equivalently, the residual satisfies $b - Ax^* = (I - P)b$, linking projections directly to regression residuals.</span>

---

## Applications in ML

* <span style="font-size: 14px;">**Linear regression**: the fitted values $\hat{y} = X(X^TX)^{-1}X^Ty$ are the projection of $y$ onto the column space of the design matrix $X$. The matrix $H = X(X^TX)^{-1}X^T$ is called the hat matrix because it "puts the hat on $y$."</span>
* <span style="font-size: 14px;">**Leverage scores**: the diagonal entries $h_{ii}$ of the hat matrix measure how influential each data point is. High leverage points ($h_{ii}$ close to 1) disproportionately affect the regression fit and deserve careful inspection.</span>
* <span style="font-size: 14px;">**Residual analysis**: the residual vector $e = (I - H)y$ is the projection onto the orthogonal complement, and $\|e\|^2$ is the sum of squared residuals (SSR). Note that $\mathrm{rank}(I - H) = m - n$ gives the residual degrees of freedom.</span>
* <span style="font-size: 14px;">**Signal processing**: projection filters extract signal components lying in a known subspace while rejecting noise in the orthogonal complement. Matched filtering and beamforming are examples.</span>
* <span style="font-size: 14px;">**Gram-Schmidt**: the Gram-Schmidt process builds an orthonormal basis by repeatedly subtracting projections onto already-found basis vectors, which is the computational core of QR decomposition.</span>
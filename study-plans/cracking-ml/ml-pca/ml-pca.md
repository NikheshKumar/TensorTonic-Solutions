# <span style="font-size: 20px;">PCA from Scratch</span>

<span style="font-size: 14px;">Principal Component Analysis (PCA) is a linear dimensionality reduction technique that transforms data into a new coordinate system where the axes (principal components) are ordered by the amount of variance they capture.</span>

---

## <span style="font-size: 16px;">Algorithm</span>

1. <span style="font-size: 14px;">**Center**: subtract the mean of each feature</span>
2. <span style="font-size: 14px;">**Covariance**: compute the</span> $d \times d$ <span style="font-size: 14px;">covariance matrix</span> $\Sigma = \frac{1}{n-1} X_c^\top X_c$
3. <span style="font-size: 14px;">**Eigendecompose**: find eigenvalues</span> $\lambda_1 \geq \lambda_2 \geq \ldots \geq \lambda_d$ <span style="font-size: 14px;">and eigenvectors</span> $\mathbf{v}_1, \ldots, \mathbf{v}_d$
4. <span style="font-size: 14px;">**Select**: keep the top</span> $k$ <span style="font-size: 14px;">eigenvectors (principal components)</span>
5. <span style="font-size: 14px;">**Project**: transform data as</span> $Z = X_c W$ <span style="font-size: 14px;">where</span> $W = [\mathbf{v}_1, \ldots, \mathbf{v}_k]$

---

## <span style="font-size: 16px;">Why Covariance Eigenvectors?</span>

<span style="font-size: 14px;">The first principal component is the direction that maximizes the variance of the projected data. This is equivalent to finding the eigenvector of the covariance matrix with the largest eigenvalue. Each subsequent component maximizes variance subject to being orthogonal to all previous components.</span>

$$
\begin{aligned}
\mathbf{v}_1 &= \arg\max_{\|\mathbf{v}\|=1} \text{Var}(X_c \mathbf{v}) \\
&= \arg\max_{\|\mathbf{v}\|=1} \mathbf{v}^{\top} \Sigma \mathbf{v}
\end{aligned}
$$

---

## <span style="font-size: 16px;">Explained Variance</span>

<span style="font-size: 14px;">The fraction of total variance explained by component</span> $i$ <span style="font-size: 14px;">is:</span>

$$
\text{EVR}_i = \frac{\lambda_i}{\sum_{j=1}^{d} \lambda_j}
$$

<span style="font-size: 14px;">This tells us how much information each component captures. A common heuristic is to keep enough components to explain 95% of the variance.</span>

---

## <span style="font-size: 16px;">eigh vs. eig</span>

<span style="font-size: 14px;">Since covariance matrices are symmetric positive semi-definite, we use</span> `np.linalg.eigh` <span style="font-size: 14px;">which is both faster and more numerically stable than the general</span> `np.linalg.eig`<span style="font-size: 14px;">. It always returns real eigenvalues and orthonormal eigenvectors.</span>

---

## <span style="font-size: 16px;">SVD Alternative</span>

<span style="font-size: 14px;">PCA can also be computed via SVD of the centered data:</span> $X_c = U S V^\top$<span style="font-size: 14px;">. The columns of</span> $V$ <span style="font-size: 14px;">are the principal components, and the singular values $s_i$ relate to eigenvalues by</span> $\lambda_i = s_i^2 / (n-1)$<span style="font-size: 14px;">. SVD is preferred for high-dimensional data where</span> $d \gg n$ <span style="font-size: 14px;">because it avoids forming the</span> $d \times d$ <span style="font-size: 14px;">covariance matrix.</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

- <span style="font-size: 14px;">**Q: When should you standardize before PCA?**</span>
  <span style="font-size: 14px;">A: When features have different units or scales. Without standardization, features with larger variance dominate the principal components.</span>

- <span style="font-size: 14px;">**Q: Can PCA handle nonlinear relationships?**</span>
  <span style="font-size: 14px;">A: Standard PCA is linear. Kernel PCA applies a kernel trick to capture nonlinear structure. Autoencoders are another nonlinear alternative.</span>

- <span style="font-size: 14px;">**Q: What is the connection to SVD?**</span>
  <span style="font-size: 14px;">A: The right singular vectors of the centered data matrix are the principal components. SVD is often used in practice because it is more numerically stable for large datasets.</span>

---
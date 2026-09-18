# <span style="font-size: 20px;">K-Means Clustering</span>

<span style="font-size: 14px;">K-Means is one of the most widely used clustering algorithms. It partitions</span> $n$ <span style="font-size: 14px;">data points into</span> $k$ <span style="font-size: 14px;">clusters by minimizing the within-cluster sum of squared distances from each point to its cluster centroid.</span>

---

## <span style="font-size: 16px;">Algorithm (Lloyd's Algorithm)</span>

1. <span style="font-size: 14px;">**Initialize**: select</span> $k$ <span style="font-size: 14px;">initial centroids (e.g., random data points)</span>
2. <span style="font-size: 14px;">**Assign**: assign each point to the nearest centroid</span>
3. <span style="font-size: 14px;">**Update**: recompute each centroid as the mean of its assigned points</span>
4. <span style="font-size: 14px;">**Repeat** steps 2-3 until centroids stop changing or max iterations is reached</span>

<span style="font-size: 14px;">Each iteration monotonically decreases the objective. Convergence to a local minimum is guaranteed, but the solution depends on initialization.</span>

---

## <span style="font-size: 16px;">Objective Function</span>

$$
J = \sum_{j=1}^{k} \sum_{\mathbf{x} \in C_j} \|\mathbf{x} - \boldsymbol{\mu}_j\|^2
$$

<span style="font-size: 14px;">where</span> $C_j$ <span style="font-size: 14px;">is the set of points in cluster</span> $j$ <span style="font-size: 14px;">and</span> $\boldsymbol{\mu}_j = \frac{1}{|C_j|}\sum_{\mathbf{x} \in C_j} \mathbf{x}$ <span style="font-size: 14px;">is the centroid.</span>

---

## <span style="font-size: 16px;">Why Mean Minimizes SSE</span>

<span style="font-size: 14px;">The mean is the point that minimizes the sum of squared Euclidean distances to all points in a set. This is why the update step uses the mean - it is the optimal centroid for fixed assignments.</span>

---

## <span style="font-size: 16px;">Initialization Matters</span>

- <span style="font-size: 14px;">Random initialization can lead to poor local minima</span>
- <span style="font-size: 14px;">K-Means++ initialization selects initial centroids to be spread apart, leading to better results</span>
- <span style="font-size: 14px;">Common practice: run K-Means multiple times with different seeds and keep the best result</span>

---

## <span style="font-size: 16px;">Choosing k</span>

- <span style="font-size: 14px;">**Elbow method**: plot total within-cluster SSE vs. k, look for an "elbow"</span>
- <span style="font-size: 14px;">**Silhouette score**: measures how similar a point is to its own cluster vs. neighboring clusters</span>
- <span style="font-size: 14px;">**Gap statistic**: compares within-cluster dispersion to a reference null distribution</span>

---

## <span style="font-size: 16px;">Limitations</span>

- <span style="font-size: 14px;">Assumes spherical, equally-sized clusters</span>
- <span style="font-size: 14px;">Sensitive to outliers (they pull centroids)</span>
- <span style="font-size: 14px;">Cannot discover non-convex clusters</span>
- <span style="font-size: 14px;">Requires specifying</span> $k$ <span style="font-size: 14px;">in advance</span>
- <span style="font-size: 14px;">Converges to local minima, not necessarily global</span>

---

## <span style="font-size: 16px;">Complexity</span>

<span style="font-size: 14px;">Each iteration costs</span> $O(n \cdot k \cdot d)$ <span style="font-size: 14px;">for computing distances. Total cost is</span> $O(T \cdot n \cdot k \cdot d)$ <span style="font-size: 14px;">where</span> $T$ <span style="font-size: 14px;">is the number of iterations until convergence.</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

- <span style="font-size: 14px;">**Q: How do you handle empty clusters?**</span>
  <span style="font-size: 14px;">A: Keep the old centroid, or reinitialize it to a random data point. Our implementation keeps the old centroid.</span>

- <span style="font-size: 14px;">**Q: When would K-Means fail?**</span>
  <span style="font-size: 14px;">A: Non-spherical clusters, clusters of very different sizes or densities, or data with many outliers.</span>

- <span style="font-size: 14px;">**Q: What is K-Medoids?**</span>
  <span style="font-size: 14px;">A: Uses actual data points (medoids) as centers instead of computed means. More robust to outliers because the center is always a real data point, but more expensive since all points must be evaluated as potential medoids.</span>

---
## What is Pearson Correlation?

Pearson correlation measures the linear relationship between two variables, producing a value between -1 and +1. A correlation of +1 indicates perfect positive linear relationship, -1 indicates perfect negative linear relationship, and 0 indicates no linear relationship. The Pearson correlation matrix computes correlations between all pairs of features in a dataset.

---

## Why Correlation Matters

**Feature selection**: Highly correlated features provide redundant information. Removing one can reduce dimensionality without losing predictive power.

**Understanding relationships**: Correlation reveals which variables move together, informing feature engineering and model interpretation.

**Assumption checking**: Many statistical methods assume uncorrelated errors or features.

**Scale invariance**: Unlike covariance, correlation is normalized and comparable across different measurement scales.

---

## The Pearson Correlation Formula

For two variables $X_i$ and $X_j$ with $N$ observations:

$$
\rho_{ij} = \frac{\text{Cov}(X_i, X_j)}{\sigma_i \sigma_j}
$$

Where:
- $\text{Cov}(X_i, X_j)$ is the covariance between $X_i$ and $X_j$
- $\sigma_i$ is the standard deviation of $X_i$
- $\sigma_j$ is the standard deviation of $X_j$

**Expanded form**:

$$
\rho_{ij} = \frac{\sum_{k=1}^{N} (x_{ki} - \bar{x}_i)(x_{kj} - \bar{x}_j)}{\sqrt{\sum_{k=1}^{N} (x_{ki} - \bar{x}_i)^2} \cdot \sqrt{\sum_{k=1}^{N} (x_{kj} - \bar{x}_j)^2}}
$$

---

## Matrix Form

For a dataset with $D$ features, the correlation matrix $R$ is $D \times D$:

$$
R = \frac{\Sigma}{\sigma \sigma^T}
$$

Where:
- $\Sigma$ is the covariance matrix
- $\sigma$ is the column vector of standard deviations
- $\sigma \sigma^T$ is the outer product creating a matrix of $\sigma_i \cdot \sigma_j$ values

**Element-wise**:

$$
R_{ij} = \frac{\Sigma_{ij}}{\sigma_i \cdot \sigma_j}
$$

---

## Properties of the Correlation Matrix

**Symmetry**: $R_{ij} = R_{ji}$ (correlation between $X_i$ and $X_j$ equals correlation between $X_j$ and $X_i$)

**Diagonal is 1**: $R_{ii} = 1$ (every variable is perfectly correlated with itself)

**Bounded values**: $-1 \leq R_{ij} \leq 1$ for all $i, j$

**Positive semi-definite**: All eigenvalues are non-negative

---

## Interpreting Correlation Values

**Strong positive** ($\rho > 0.7$): As one variable increases, the other tends to increase substantially

**Moderate positive** ($0.3 < \rho < 0.7$): Positive relationship but with considerable scatter

**Weak/No correlation** ($-0.3 < \rho < 0.3$): Little to no linear relationship

**Moderate negative** ($-0.7 < \rho < -0.3$): Negative relationship with scatter

**Strong negative** ($\rho < -0.7$): As one variable increases, the other tends to decrease substantially

---

## Worked Example

**Dataset** (4 samples, 2 features):
- Sample 0: X = 1, Y = 2
- Sample 1: X = 2, Y = 4
- Sample 2: X = 3, Y = 5
- Sample 3: X = 4, Y = 4

**Step 1 - Calculate means**:

$$
\bar{X} = \frac{1 + 2 + 3 + 4}{4} = 2.5
$$

$$
\bar{Y} = \frac{2 + 4 + 5 + 4}{4} = 3.75
$$

**Step 2 - Calculate deviations**:

- $(X - \bar{X})$: [-1.5, -0.5, 0.5, 1.5]
- $(Y - \bar{Y})$: [-1.75, 0.25, 1.25, 0.25]

**Step 3 - Calculate covariance**:

$$
\text{Cov}(X, Y) = \frac{(-1.5)(-1.75) + (-0.5)(0.25) + (0.5)(1.25) + (1.5)(0.25)}{3}
$$

$$
= \frac{2.625 - 0.125 + 0.625 + 0.375}{3} = \frac{3.5}{3} \approx 1.167
$$

**Step 4 - Calculate standard deviations**:

$$
\sigma_X = \sqrt{\frac{(-1.5)^2 + (-0.5)^2 + (0.5)^2 + (1.5)^2}{3}} = \sqrt{\frac{5}{3}} \approx 1.29
$$

$$
\sigma_Y = \sqrt{\frac{(-1.75)^2 + (0.25)^2 + (1.25)^2 + (0.25)^2}{3}} = \sqrt{\frac{4.75}{3}} \approx 1.26
$$

**Step 5 - Calculate correlation**:

$$
\rho_{XY} = \frac{1.167}{1.29 \times 1.26} \approx \frac{1.167}{1.625} \approx 0.72
$$

**Interpretation**: Strong positive correlation between X and Y.

---

## Handling Zero Variance Features

When a feature has zero variance (all values identical):

$$
\sigma_i = 0 \Rightarrow \rho_{ij} = \frac{\text{Cov}}{0 \cdot \sigma_j} = \text{undefined}
$$

**Standard handling**: Return NaN for any correlation involving a zero-variance feature

**Rationale**: A constant feature has no relationship with anything since it never varies

---

## Input Validation

**Minimum samples**: Correlation requires at least $N \geq 2$ samples to compute variance

**Dimensionality**: Input must be 2D (samples × features)

**Invalid inputs**: Return None for:
- $N < 2$ samples
- Non-2D arrays
- Empty arrays

---

## Vectorized Computation

To compute the full correlation matrix efficiently:

1. **Center the data**: Subtract column means
2. **Compute covariance matrix**: $\Sigma = \tilde{X}^T \tilde{X} / (N-1)$
3. **Extract standard deviations**: $\sigma = \sqrt{\text{diag}(\Sigma)}$
4. **Normalize**: $R = \Sigma / (\sigma \sigma^T)$

This avoids nested loops over feature pairs and leverages optimized matrix operations.

---

## Correlation vs Covariance

**Covariance**:
- Measures joint variability
- Scale-dependent (units matter)
- Range: $(-\infty, +\infty)$
- Hard to interpret magnitude

**Correlation**:
- Normalized covariance
- Scale-independent (dimensionless)
- Range: $[-1, +1]$
- Directly interpretable

**Relationship**: Correlation = Covariance / (Product of standard deviations)

---

## Limitations of Pearson Correlation

**Linear relationships only**: Perfect quadratic relationship ($Y = X^2$) may have zero Pearson correlation

**Sensitive to outliers**: A single extreme point can dramatically change correlation

**Not robust**: Non-normal distributions may mislead interpretation

**Alternatives**:
- Spearman correlation (rank-based, handles non-linear monotonic relationships)
- Kendall tau (also rank-based, more robust to outliers)
- Distance correlation (detects non-linear dependencies)

---

## Where Pearson Correlation Shows Up

- **Feature Selection**: Identifying and removing redundant features

- **Exploratory Data Analysis**: Understanding variable relationships before modeling

- **Finance**: Correlating asset returns for portfolio diversification

- **Genomics**: Gene expression correlation networks

- **Signal Processing**: Cross-correlation for signal alignment

- **Quality Control**: Monitoring correlation between process variables

- **Psychology**: Correlating test scores and measurements

- **Climate Science**: Correlating temperature, precipitation, and other variables

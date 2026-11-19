## What is Min-Max Scaling?

Min-Max scaling, also called normalization, transforms features to a fixed range, typically [0, 1]. The minimum value becomes 0, the maximum becomes 1, and all other values are linearly scaled between them. This ensures all features have the same scale, which is critical for many machine learning algorithms.

---

## Why Scale Features?

**Distance-based algorithms**: KNN, K-means, and SVM with RBF kernel compute distances between samples. A feature measured in thousands (e.g., salary) would dominate one measured in single digits (e.g., years of experience).

**Gradient descent optimization**: Neural networks and logistic regression converge faster when features are on similar scales. Large-scale features cause large gradients, leading to unstable training.

**Regularization fairness**: L1 and L2 penalties treat all features equally only when they are on the same scale.

**Interpretability**: After scaling, a value of 0.7 means "70% of the way from minimum to maximum" regardless of the original units.

---

## The Min-Max Scaling Formula

For a feature column $X$, each value $x$ is transformed to:

$$
x' = \frac{x - x_{min}}{x_{max} - x_{min}}
$$

Where:
- $x_{min}$ = minimum value in the feature column
- $x_{max}$ = maximum value in the feature column
- $x'$ = scaled value in [0, 1]

**Properties**:
- When $x = x_{min}$: $x' = 0$
- When $x = x_{max}$: $x' = 1$
- Linear transformation preserving relative distances

---

## Scaling to Arbitrary Range

To scale to a range $[a, b]$ instead of $[0, 1]$:

$$
x' = a + (b - a) \cdot \frac{x - x_{min}}{x_{max} - x_{min}}
$$

**Common alternative ranges**:
- $[-1, 1]$: Centered around zero, useful for algorithms that expect symmetric inputs
- $[0, 255]$: Image pixel values
- Custom ranges based on domain requirements

---

## Column-wise Application

Min-Max scaling is applied independently to each feature (column):

**Original data**:
- Feature A: [100, 200, 300, 400, 500]
- Feature B: [2, 4, 6, 8, 10]

**Compute min/max per column**:
- Feature A: min=100, max=500, range=400
- Feature B: min=2, max=10, range=8

**Scale each column**:
- Feature A: [(100-100)/400, (200-100)/400, ...] = [0, 0.25, 0.5, 0.75, 1.0]
- Feature B: [(2-2)/8, (4-2)/8, ...] = [0, 0.25, 0.5, 0.75, 1.0]

Now both features range from 0 to 1.

---

## Worked Example

**Original feature values**: [10, 20, 30, 40, 50]

**Step 1 - Find min and max**:
- $x_{min} = 10$
- $x_{max} = 50$
- Range = $50 - 10 = 40$

**Step 2 - Apply formula to each value**:

$$
x'_1 = \frac{10 - 10}{40} = 0.0
$$

$$
x'_2 = \frac{20 - 10}{40} = 0.25
$$

$$
x'_3 = \frac{30 - 10}{40} = 0.5
$$

$$
x'_4 = \frac{40 - 10}{40} = 0.75
$$

$$
x'_5 = \frac{50 - 10}{40} = 1.0
$$

**Result**: [0.0, 0.25, 0.5, 0.75, 1.0]

---

## Handling Zero Range

When all values in a feature are identical ($x_{max} = x_{min}$), the denominator is zero:

$$
x' = \frac{x - x_{min}}{0} = \text{undefined}
$$

**Solutions**:
- Set all scaled values to 0 (most common)
- Set all scaled values to 0.5 (midpoint of range)
- Remove the feature (constant features have no predictive value)

---

## Train-Test Considerations

**Critical rule**: Compute $x_{min}$ and $x_{max}$ from training data only.

**Why?** Using test data statistics causes data leakage. The model would have indirect knowledge of test data distribution, leading to overly optimistic performance estimates.

**Applying to test data**:

$$
x'_{test} = \frac{x_{test} - x_{min,train}}{x_{max,train} - x_{min,train}}
$$

**Consequence**: Test values may fall outside [0, 1] if test data has values beyond the training range.

---

## Inverse Transform

To convert scaled values back to original scale:

$$
x = x' \cdot (x_{max} - x_{min}) + x_{min}
$$

**Use cases**:
- Interpreting model predictions in original units
- Post-processing outputs for human understanding
- Reversing preprocessing for deployment

---

## Outlier Sensitivity

Min-Max scaling is highly sensitive to outliers:

**Example without outlier**: [10, 20, 30, 40, 50]
- Scaled: [0, 0.25, 0.5, 0.75, 1.0]

**Example with outlier**: [10, 20, 30, 40, 1000]
- min=10, max=1000, range=990
- Scaled: [0, 0.01, 0.02, 0.03, 1.0]

A single outlier compressed all normal values near 0. Consider:
- Removing outliers before scaling
- Using Robust Scaling (based on percentiles)
- Winsorization (capping outliers)

---

## Min-Max vs Z-Score Standardization

**Min-Max Scaling**:
- Bounded range [0, 1]
- Preserves zero values (if min is 0)
- Sensitive to outliers
- Good for: neural networks, image data, bounded algorithms

**Z-Score Standardization**:
- Unbounded range (any real number)
- Centers data at mean 0
- Less sensitive to outliers
- Good for: PCA, linear regression, SVM

---

## Where Min-Max Scaling Shows Up

- **Neural Networks**: Input normalization for stable training

- **Image Processing**: Pixel values scaled to [0, 1] from [0, 255]

- **K-Nearest Neighbors**: Distance calculations require comparable scales

- **Support Vector Machines**: Kernel methods sensitive to feature magnitudes

- **Visualization**: Heatmaps and color scales with fixed ranges

- **Feature Engineering**: Combining features from different sources

- **Time Series**: Normalizing different series for comparison

- **Clustering**: K-means and hierarchical clustering use distances

- **Anomaly Detection**: Isolation Forest and LOF benefit from scaled features

- **Gradient Boosting**: While tree-based models are scale-invariant, scaling can help with convergence

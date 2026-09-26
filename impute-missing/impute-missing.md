## What is Missing Data Imputation?

Missing data imputation is the process of replacing missing or null values in a dataset with substituted values. Real-world datasets almost always contain missing entries due to sensor failures, survey non-responses, data entry errors, or intentional omissions. Imputation allows analysis to proceed on complete datasets while preserving as much information as possible.

---

## Why Missing Data is Problematic

**Algorithm requirements**: Many machine learning algorithms cannot handle missing values and will fail or produce errors.

**Reduced sample size**: Deleting rows with missing values (listwise deletion) can dramatically reduce the dataset, losing valuable information.

**Biased results**: If data is not missing randomly, deletion can introduce systematic bias.

**Feature loss**: Deleting columns with missing values loses entire features that might be predictive.

---

## Types of Missing Data Mechanisms

Understanding why data is missing determines the appropriate imputation strategy:

**Missing Completely at Random (MCAR)**:
- Missingness is unrelated to any variables
- Example: A sensor randomly fails due to power fluctuations
- Safe to delete or impute with simple methods
- Rarest in practice

**Missing at Random (MAR)**:
- Missingness depends on observed variables but not the missing value itself
- Example: Younger people are less likely to report income (age observed, income missing)
- Can be handled with methods that use other features to predict missing values

**Missing Not at Random (MNAR)**:
- Missingness depends on the missing value itself
- Example: High earners refuse to report income precisely because it is high
- Most challenging - requires domain knowledge or specialized models

---

## Simple Imputation Methods

### Mean Imputation

Replace missing values with the mean of the observed values for that feature.

$$
x_{imputed} = \bar{x} = \frac{1}{n_{observed}} \sum_{i \in observed} x_i
$$

**Advantages**:
- Simple and fast
- Preserves the mean of the feature

**Disadvantages**:
- Reduces variance (imputed values cluster at mean)
- Distorts distributions and correlations
- Ignores relationships between features

---

### Median Imputation

Replace missing values with the median of the observed values.

**Advantages**:
- Robust to outliers (unlike mean)
- Appropriate for skewed distributions

**Disadvantages**:
- Same issues as mean imputation regarding variance reduction
- Does not use information from other features

---

### Mode Imputation

Replace missing categorical values with the most frequent category.

**Advantages**:
- Simple approach for categorical data
- Preserves the most common category

**Disadvantages**:
- Can overrepresent the dominant category
- Ignores category distributions and relationships

---

### Constant Imputation

Replace missing values with a fixed constant (e.g., 0, -1, or a special indicator).

**Use cases**:
- When missingness itself is informative
- Creating a "missing" category for categorical variables
- When domain knowledge suggests a natural default

**Considerations**:
- The constant should not be confused with real data values
- Often combined with a missingness indicator feature

---

## Advanced Imputation Methods

### K-Nearest Neighbors (KNN) Imputation

Find the $k$ most similar samples (using observed features) and impute using their values.

For numeric features:

$$
x_{imputed} = \frac{1}{k} \sum_{j \in neighbors} x_j
$$

For categorical features, use the mode (most common value) among neighbors.

**Advantages**:
- Uses relationships between samples
- No parametric assumptions

**Disadvantages**:
- Computationally expensive for large datasets
- Sensitive to choice of $k$ and distance metric

---

### Multiple Imputation

Generate multiple complete datasets, each with different imputed values, analyze each, and pool results.

**Process**:
1. Create $m$ imputed datasets (typically 5-20)
2. Perform analysis on each dataset
3. Combine results using Rubin's rules to account for imputation uncertainty

**Advantages**:
- Properly accounts for uncertainty in imputed values
- Gold standard for statistical inference

**Disadvantages**:
- More complex to implement
- Requires running analysis multiple times

---

### Regression Imputation

Predict missing values using a regression model trained on complete cases.

$$
\hat{x}_{missing} = \beta_0 + \beta_1 z_1 + \beta_2 z_2 + ... + \beta_p z_p
$$

Where $z_1, ..., z_p$ are other observed features.

**Advantages**:
- Uses information from correlated features
- Can capture complex relationships with non-linear models

**Disadvantages**:
- Imputed values have no residual variance (too precise)
- Model misspecification affects imputation quality

---

## Worked Example: Mean Imputation

**Dataset** (feature values with missing entry):

Sample 1: Age=25, Income=50000
Sample 2: Age=30, Income=NaN (missing)
Sample 3: Age=35, Income=70000
Sample 4: Age=40, Income=80000

**Step 1 - Calculate mean of observed Income values**:

$$
\bar{Income} = \frac{50000 + 70000 + 80000}{3} = 66667
$$

**Step 2 - Replace missing value**:
Sample 2 Income becomes 66667

**Result**:
Sample 1: Age=25, Income=50000
Sample 2: Age=30, Income=66667
Sample 3: Age=35, Income=70000
Sample 4: Age=40, Income=80000

---

## Worked Example: KNN Imputation

**Same dataset, using k=2 nearest neighbors based on Age**:

**Step 1 - Find 2 nearest neighbors to Sample 2 (Age=30)**:
- Sample 1: Age=25, distance = |30-25| = 5
- Sample 3: Age=35, distance = |30-35| = 5
- Sample 4: Age=40, distance = |30-40| = 10

Nearest neighbors: Samples 1 and 3 (both have distance 5)

**Step 2 - Impute using neighbors' Income values**:

$$
Income_{imputed} = \frac{50000 + 70000}{2} = 60000
$$

**Result**: Different from mean imputation because it uses local information based on similar samples.

---

## Handling Multiple Missing Features

When a sample has multiple missing values, strategies include:

- **Sequential imputation**: Impute one feature at a time, using previously imputed values
- **Joint imputation**: Model all missing values simultaneously
- **Iterative imputation**: Cycle through features multiple times until convergence (MICE algorithm)

---

## Creating Missingness Indicators

In addition to imputing, create binary features indicating whether values were originally missing:

$$
\text{Income\_missing} = \begin{cases} 1 & \text{if Income was NaN} \\ 0 & \text{otherwise} \end{cases}
$$

**Why useful**:
- Missingness itself might be predictive (informative missingness)
- Allows models to learn different patterns for imputed vs observed values
- Preserves information that would otherwise be lost

---

## Where Missing Data Imputation Shows Up

- **Clinical Data**: Patient records with missing lab tests, medications, or outcomes

- **Survey Analysis**: Non-response to sensitive questions about income, health, or behavior

- **Sensor Networks**: IoT devices with intermittent connectivity or failures

- **Financial Data**: Missing historical prices, unreported transactions

- **Recommendation Systems**: Sparse user-item matrices where most entries are missing

- **Time Series**: Missing observations due to equipment downtime or reporting delays

- **Natural Language Processing**: Incomplete metadata, missing labels for semi-supervised learning

- **Image Processing**: Missing pixels, corrupted regions, incomplete scans

- **Genomics**: Missing genotypes due to sequencing quality issues

- **Social Network Analysis**: Missing edges or node attributes

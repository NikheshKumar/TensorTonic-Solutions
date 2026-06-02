# <span style="font-size: 20px;">Missing Value Imputation</span>

<span style="font-size: 14px;">Missing data is one of the most common problems in real-world machine learning. Sensors malfunction, survey respondents skip questions, and database records are incomplete. How you handle missing values can dramatically affect model performance. Understanding imputation strategies is essential for data preprocessing interviews and practical ML engineering.</span>

---

## <span style="font-size: 16px;">Why Missing Data Matters</span>

<span style="font-size: 14px;">Most machine learning algorithms cannot handle NaN values directly. Feeding missing data into a model will either cause errors or produce meaningless results. Before training, you must decide how to deal with these gaps.</span>

<span style="font-size: 14px;">There are three categories of missing data:</span>

<span style="font-size: 14px;">**MCAR (Missing Completely At Random)**: the probability of a value being missing is independent of both observed and unobserved data. Example: a sensor randomly fails due to power fluctuations.</span>

<span style="font-size: 14px;">**MAR (Missing At Random)**: the probability of a value being missing depends on other observed variables, but not on the missing value itself. Example: younger respondents are less likely to report income, but the missingness does not depend on the actual income value.</span>

<span style="font-size: 14px;">**MNAR (Missing Not At Random)**: the probability of a value being missing depends on the missing value itself. Example: high-income earners are less likely to report their income. This is the hardest case to handle correctly.</span>

<span style="font-size: 14px;">Simple imputation methods (mean, median, mode) work well for MCAR data but can introduce bias for MAR or MNAR data. More sophisticated methods like KNN or multiple imputation can partially address these issues.</span>

---

## <span style="font-size: 16px;">Mean and Median Imputation</span>

<span style="font-size: 14px;">**Mean imputation** replaces each missing value with the average of all observed values in the same column:</span>

$$
\hat{X}_{ij} = \bar{X}_j = \frac{1}{|\mathcal{O}_j|} \sum_{k \in \mathcal{O}_j} X_{kj}
$$

<span style="font-size: 14px;">where $\mathcal{O}_j$ is the set of row indices with observed (non-NaN) values in column $j$.</span>

<span style="font-size: 14px;">**Median imputation** replaces missing values with the column median instead. This is more robust to outliers. If column $j$ has non-NaN values sorted as $x_{(1)} \leq x_{(2)} \leq \cdots \leq x_{(m)}$, the median is $x_{(\lceil m/2 \rceil)}$ for odd $m$, or the average of $x_{(m/2)}$ and $x_{(m/2+1)}$ for even $m$.</span>

<span style="font-size: 14px;">**Advantages**: simple, fast, preserves the column mean (or median). **Disadvantages**: reduces variance, distorts correlations between features, and does not account for relationships between variables.</span>

---

## <span style="font-size: 16px;">Most Frequent (Mode) Imputation</span>

<span style="font-size: 14px;">Mode imputation replaces missing values with the most frequently occurring non-NaN value in the column. This is particularly useful for categorical features or discrete-valued columns where mean/median may not produce valid values.</span>

<span style="font-size: 14px;">For a column with non-NaN values $\{v_1, v_2, \ldots, v_m\}$:</span>

$$
\hat{X}_{ij} = \arg\max_{v} \sum_{k \in \mathcal{O}_j} \mathbf{1}[X_{kj} = v]
$$

<span style="font-size: 14px;">When there is a tie (multiple values share the highest frequency), the implementation should return any valid mode. A common convention is to return the smallest value among the tied modes.</span>

<span style="font-size: 14px;">**Advantages**: works for both numerical and categorical data. **Disadvantages**: can heavily bias the distribution toward the mode, especially when many values are missing.</span>

---

## <span style="font-size: 16px;">KNN Imputation</span>

<span style="font-size: 14px;">KNN imputation is a more sophisticated approach that considers the relationships between samples. For each missing value $X_{ij}$:</span>

<span style="font-size: 14px;">1. Find rows that have an observed value in column $j$.</span>
<span style="font-size: 14px;">2. Compute the Euclidean distance between row $i$ and each candidate row, using only features where **both** rows have observed values.</span>
<span style="font-size: 14px;">3. Select the $k$ nearest neighbors.</span>
<span style="font-size: 14px;">4. Impute with the mean of those neighbors' values in column $j$.</span>

$$
d(i, m) = \sqrt{\sum_{l \in S_{im}} (X_{il} - X_{ml})^2}
$$

<span style="font-size: 14px;">where $S_{im}$ is the set of features where both rows $i$ and $m$ have non-NaN values. If no shared features exist or no valid neighbors can be found, the method falls back to the column mean.</span>

<span style="font-size: 14px;">**Advantages**: captures local structure and correlations between features. **Disadvantages**: computationally expensive ($O(n^2 \cdot d)$ per missing value), sensitive to the choice of $k$, and requires careful handling of the distance metric when many features are missing.</span>

---

## <span style="font-size: 16px;">Practical Considerations</span>

<span style="font-size: 14px;">**Feature scaling**: KNN imputation is sensitive to feature scales. In practice, you should normalize or standardize features before computing distances. For this problem, we assume features are on comparable scales.</span>

<span style="font-size: 14px;">**Missing indicator features**: a common practice is to create binary indicator columns that flag which values were originally missing. This allows the model to learn patterns related to missingness itself.</span>

<span style="font-size: 14px;">**Multiple imputation**: instead of producing a single imputed dataset, generate multiple imputed versions and combine model results. This better captures the uncertainty introduced by imputation.</span>

<span style="font-size: 14px;">**Iterative imputation (MICE)**: impute each feature iteratively using a regression model conditioned on all other features. This captures inter-feature dependencies better than column-wise methods.</span>

<span style="font-size: 14px;">**When to drop instead of impute**: if a feature has more than 50-70% missing values, imputation may introduce more noise than signal. Similarly, if a sample has most features missing, dropping that sample may be preferable.</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

- <span style="font-size: 14px;">**Q: When would you choose median imputation over mean imputation?**</span>
  <span style="font-size: 14px;">A: Median imputation is preferred when the column distribution is skewed or contains outliers. The mean is sensitive to extreme values, so a few outliers can pull the imputed values away from the center of the distribution. For example, if income data has a few very high earners, mean imputation would fill missing values with an inflated estimate, while median imputation would use a more representative central value.</span>

- <span style="font-size: 14px;">**Q: What are the risks of imputing missing values before splitting into train/test sets?**</span>
  <span style="font-size: 14px;">A: This causes data leakage. If you compute column means (or medians, modes, or KNN distances) using the entire dataset including the test set, the imputed training values contain information from test samples. The correct approach is to fit the imputation statistics on the training set only, then transform both training and test sets using those statistics. Scikit-learn's Pipeline and ColumnTransformer enforce this correctly.</span>

- <span style="font-size: 14px;">**Q: How does the choice of $k$ in KNN imputation affect the result?**</span>
  <span style="font-size: 14px;">A: Small $k$ (e.g., 1-3) makes the imputation sensitive to local noise - a single unusual neighbor can heavily influence the imputed value. Large $k$ produces smoother imputations that approach the column mean as $k$ increases, losing the benefit of local structure. Cross-validation on a downstream task can help select an appropriate $k$. A common default is $k = 5$.</span>

- <span style="font-size: 14px;">**Q: How would you handle missing values in a categorical feature?**</span>
  <span style="font-size: 14px;">A: Mode imputation is the simplest approach for categorical features. More sophisticated methods include: (1) treating "missing" as its own category, which works well when missingness itself is informative; (2) using a classifier (e.g., random forest) trained on other features to predict the missing category; (3) using entity embeddings where the missing category gets its own learned embedding vector.</span>

- <span style="font-size: 14px;">**Q: What is the difference between single imputation and multiple imputation?**</span>
  <span style="font-size: 14px;">A: Single imputation produces one complete dataset and treats imputed values as if they were observed, underestimating uncertainty. Multiple imputation (e.g., MICE) generates $m$ complete datasets, each with slightly different imputed values drawn from the predictive distribution. Models are trained on each dataset separately and results are pooled using Rubin's rules, which correctly accounts for the additional uncertainty due to missing data. Multiple imputation is the gold standard in statistics but adds computational cost.</span>

---
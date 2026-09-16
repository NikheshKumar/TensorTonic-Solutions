# <span style="font-size: 20px;">Categorical Encoding</span>

<span style="font-size: 14px;">Categorical encoding transforms non-numeric features into numerical representations that machine learning algorithms can process. The choice of encoding method depends on the type of categorical variable and the algorithm being used.</span>

---

## <span style="font-size: 16px;">Why Encode Categorical Variables?</span>

<span style="font-size: 14px;">Most ML algorithms operate on numerical data. A feature like "color" with values {"red", "green", "blue"} has no inherent numerical meaning. Encoding assigns numbers to these categories in a principled way.</span>

<span style="font-size: 14px;">There are two main types of categorical variables:</span>

<span style="font-size: 14px;">**Nominal**: categories with no inherent order (e.g., color, country, blood type). One-hot encoding is typically preferred here because assigning arbitrary integers would imply a false ordering.</span>

<span style="font-size: 14px;">**Ordinal**: categories with a meaningful order (e.g., education level: high school < bachelor < master < PhD). Ordinal encoding preserves this ordering, and label encoding can also work if the sort order is meaningful.</span>

---

## <span style="font-size: 16px;">Label Encoding</span>

<span style="font-size: 14px;">Label encoding maps each unique category to an integer. The standard approach sorts unique values alphabetically and assigns indices 0, 1, 2, and so on.</span>

$$
\text{mapping} = \{c_0: 0, c_1: 1, \ldots, c_{k-1}: k-1\}
$$

<span style="font-size: 14px;">where</span> $c_0 < c_1 < \ldots < c_{k-1}$ <span style="font-size: 14px;">in alphabetical order.</span>

<span style="font-size: 14px;">Label encoding is compact (single column per feature) but introduces an artificial ordinal relationship. For tree-based models (decision trees, random forests, gradient boosting) this is usually acceptable because they split on thresholds. For linear models and neural networks, the implied ordering can mislead the model.</span>

---

## <span style="font-size: 16px;">One-Hot Encoding</span>

<span style="font-size: 14px;">One-hot encoding creates a binary vector of length</span> $C$ <span style="font-size: 14px;">(number of classes) for each sample. The vector has a 1 at the position corresponding to the label and 0 everywhere else:</span>

$$
\text{one\_hot}(k, C) = [\underbrace{0, \ldots, 0}_{k}, 1, \underbrace{0, \ldots, 0}_{C-k-1}]
$$

<span style="font-size: 14px;">This avoids imposing any ordinal relationship between categories. Each category becomes an independent binary feature. The tradeoff is dimensionality: a feature with 1000 unique values becomes 1000 binary columns. This is sometimes called the "dummy variable trap" when all columns are kept in linear models, since they sum to 1 (perfect multicollinearity).</span>

---

## <span style="font-size: 16px;">Ordinal Encoding</span>

<span style="font-size: 14px;">Ordinal encoding maps categories to integers according to a user-specified ordering rather than alphabetical order. This preserves the natural hierarchy of ordinal variables.</span>

$$
\text{order} = [c_0, c_1, \ldots, c_{k-1}] \implies c_i \mapsto i
$$

<span style="font-size: 14px;">For example, encoding education levels as ["high_school", "bachelor", "master", "phd"] maps to [0, 1, 2, 3], preserving the progression. Using alphabetical label encoding would produce ["bachelor":0, "high_school":1, "master":2, "phd":3], which incorrectly places "high_school" above "bachelor".</span>

---

## <span style="font-size: 16px;">When to Use Which Encoding</span>

<span style="font-size: 14px;">**Label encoding** is best for target variables in classification (e.g., scikit-learn's LabelEncoder), ordinal features with alphabetical ordering, or tree-based models where split thresholds handle the artificial ordering.</span>

<span style="font-size: 14px;">**One-hot encoding** is best for nominal features with linear models, neural networks, or any algorithm sensitive to ordinal relationships. Avoid it when cardinality is very high (thousands of unique values) as it creates sparse, high-dimensional data.</span>

<span style="font-size: 14px;">**Ordinal encoding** is best when there is a known, meaningful order to categories that does not match alphabetical sorting. Common examples: severity levels (low/medium/high), size categories (S/M/L/XL), education levels.</span>

<span style="font-size: 14px;">Beyond these three, more advanced encodings exist: target encoding (replace category with mean target value), frequency encoding (replace with occurrence count), binary encoding (binary representation of label codes), and embedding layers in neural networks.</span>

---

## <span style="font-size: 16px;">Common Pitfalls</span>

<span style="font-size: 14px;">**Data leakage with target encoding**: computing target statistics on the full dataset (including test) leaks information. Always fit encodings on training data only.</span>

<span style="font-size: 14px;">**Unseen categories at test time**: if a category appears in test data but not in training, the encoder has no mapping. Common strategies are to assign a default "unknown" code or to use a hash-based encoder.</span>

<span style="font-size: 14px;">**High cardinality**: one-hot encoding a feature with thousands of values creates a very sparse matrix. This increases memory usage, training time, and can hurt model performance. Consider target encoding or embedding layers instead.</span>

<span style="font-size: 14px;">**Multicollinearity**: in linear regression, keeping all one-hot columns creates perfect multicollinearity (they sum to 1). Dropping one column (the "reference" category) solves this.</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

- <span style="font-size: 14px;">**Q: When would label encoding be harmful for a linear model?**</span>
  <span style="font-size: 14px;">A: Label encoding assigns arbitrary integers to nominal categories, implying an ordering that does not exist. A linear model would learn that "category 3" is three times "category 1", which is meaningless for nominal data. This biases the learned weights. One-hot encoding avoids this by treating each category independently.</span>

- <span style="font-size: 14px;">**Q: How do you handle a categorical feature with 10,000 unique values?**</span>
  <span style="font-size: 14px;">A: One-hot encoding would create 10,000 columns, which is impractical. Better options include target encoding (replace category with smoothed target mean), frequency encoding, hashing trick (hash categories to a fixed number of buckets), or learned embeddings in neural networks.</span>

- <span style="font-size: 14px;">**Q: What is the dummy variable trap?**</span>
  <span style="font-size: 14px;">A: When all one-hot columns are included in a linear model, they sum to 1 for every row, creating perfect multicollinearity with the intercept term. This makes the normal equation singular. The fix is to drop one column (called "drop first" or the reference category), reducing the encoding from C columns to C-1.</span>

- <span style="font-size: 14px;">**Q: How would you encode categories for a gradient boosting model?**</span>
  <span style="font-size: 14px;">A: Gradient boosting (XGBoost, LightGBM, CatBoost) uses decision tree splits, so label/ordinal encoding works well because the tree can split at any threshold. LightGBM and CatBoost have native categorical feature support that avoids encoding altogether. One-hot encoding is usually worse for tree models because it creates many sparse binary features, reducing split quality.</span>

- <span style="font-size: 14px;">**Q: What is target encoding and when is it risky?**</span>
  <span style="font-size: 14px;">A: Target encoding replaces each category with the mean (or other statistic) of the target variable for that category. It is powerful for high-cardinality features but risks overfitting and data leakage. Categories seen only once get their exact target value, giving the model a near-perfect feature. Mitigation strategies include leave-one-out encoding, smoothing with the global mean, and always fitting only on training data.</span>

---
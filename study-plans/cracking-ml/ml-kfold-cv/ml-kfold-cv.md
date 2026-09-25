# <span style="font-size: 20px;">K-Fold Cross-Validation</span>

<span style="font-size: 14px;">K-Fold Cross-Validation is the standard method for estimating how well a model will generalize to unseen data. It provides a more reliable estimate than a single train/test split by using every data point for both training and validation.</span>

---

## <span style="font-size: 16px;">Algorithm</span>

1. <span style="font-size: 14px;">Shuffle the data randomly</span>
2. <span style="font-size: 14px;">Split into</span> $k$ <span style="font-size: 14px;">approximately equal-sized folds</span>
3. <span style="font-size: 14px;">For each fold</span> $i = 1, \ldots, k$<span style="font-size: 14px;">:</span>
   - <span style="font-size: 14px;">Use fold</span> $i$ <span style="font-size: 14px;">as the validation set</span>
   - <span style="font-size: 14px;">Use the remaining</span> $k-1$ <span style="font-size: 14px;">folds as the training set</span>
   - <span style="font-size: 14px;">Train the model and evaluate on the validation fold</span>
4. <span style="font-size: 14px;">Report the per-fold scores and the mean of the</span> $k$ <span style="font-size: 14px;">scores</span>

---

## <span style="font-size: 16px;">Choosing k</span>

- $k = 5$ <span style="font-size: 14px;">or</span> $k = 10$<span style="font-size: 14px;">: most common choices, good balance of bias and variance</span>
- $k = n$ <span style="font-size: 14px;">(Leave-One-Out): low bias but high variance, expensive for large datasets</span>
- $k = 2$<span style="font-size: 14px;">: high bias (each training set is only half the data)</span>

---

## <span style="font-size: 16px;">Bias-Variance Tradeoff in CV</span>

- <span style="font-size: 14px;">**Large k**: each training set is nearly the full dataset, so bias is low. But training sets overlap heavily, making estimates correlated and increasing variance</span>
- <span style="font-size: 14px;">**Small k**: each training set is smaller, introducing bias. But folds are more independent, reducing variance</span>

---

## <span style="font-size: 16px;">Stratified K-Fold</span>

<span style="font-size: 14px;">Standard K-Fold may produce folds with imbalanced class distributions. Stratified K-Fold preserves the class distribution in each fold. This is important for imbalanced datasets.</span>

---

## <span style="font-size: 16px;">Computational Complexity</span>

- <span style="font-size: 14px;">K-Fold requires training the model $k$ times, so total cost is $O(k \cdot T)$ where $T$ is the cost of a single training run</span>
- <span style="font-size: 14px;">Leave-One-Out ($k = n$) is prohibitive for expensive models but acceptable for closed-form solutions like ridge regression, where the LOO score can be computed in $O(n)$ via the hat matrix</span>
- <span style="font-size: 14px;">Stratified splitting adds negligible overhead: it only affects how indices are assigned to folds</span>
- <span style="font-size: 14px;">For large datasets, even $k = 5$ may be expensive - consider a single holdout or progressive validation instead</span>
---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

- <span style="font-size: 14px;">**Q: Why shuffle before splitting?**</span>
  <span style="font-size: 14px;">A: Without shuffling, folds may not be representative if the data has ordering (e.g., sorted by class). Shuffling ensures each fold is a random sample.</span>

- <span style="font-size: 14px;">**Q: Can you use CV for hyperparameter tuning?**</span>
  <span style="font-size: 14px;">A: Yes, but you need nested CV: an outer loop for estimating generalization, and an inner loop for selecting hyperparameters. Using the same CV for both leads to optimistic estimates.</span>

- <span style="font-size: 14px;">**Q: What about time series?**</span>
  <span style="font-size: 14px;">A: Standard K-Fold breaks temporal order. Use time-series split instead: train on past data, validate on future data.</span>

---
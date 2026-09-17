# <span style="font-size: 20px;">Permutation Feature Importance</span>

<span style="font-size: 14px;">Permutation feature importance, introduced by Breiman for random forests and generalized by Fisher, Rudin, and Dominici, is a model-agnostic method for measuring how much a model depends on each feature. It works by randomly shuffling each feature and measuring the resulting decrease in model performance.</span>

---

## <span style="font-size: 16px;">Key Idea</span>

<span style="font-size: 14px;">If a feature is important to the model, shuffling its values will break the relationship between that feature and the target, causing the model's performance to drop. If a feature is unimportant, shuffling it will have little effect.</span>

$$
I_j = \text{score}(X, y) - \text{score}(X^{(j)}, y)
$$

<span style="font-size: 14px;">where</span> $X^{(j)}$ <span style="font-size: 14px;">is the data with feature</span> $j$ <span style="font-size: 14px;">randomly permuted. A positive value means the feature is important; zero or negative means unimportant.</span>

---

## <span style="font-size: 16px;">Algorithm</span>

1. <span style="font-size: 14px;">Compute baseline score (e.g., accuracy) on unperturbed data</span>
2. <span style="font-size: 14px;">For each feature</span> $j = 1, \ldots, d$<span style="font-size: 14px;">:</span>
   - <span style="font-size: 14px;">For each repeat</span> $r = 1, \ldots, R$<span style="font-size: 14px;">:</span>
     - <span style="font-size: 14px;">Copy the data and randomly permute column</span> $j$
     - <span style="font-size: 14px;">Compute score on the permuted data</span>
     - <span style="font-size: 14px;">Record the drop: baseline - permuted score</span>
   - <span style="font-size: 14px;">Feature</span> $j$<span style="font-size: 14px;">'s importance = mean of drops across repeats</span>

<span style="font-size: 14px;">Multiple repeats reduce variance in the importance estimate due to randomness in permutation.</span>

---

## <span style="font-size: 16px;">Why Permutation?</span>

<span style="font-size: 14px;">Permuting a feature breaks its relationship with the target while preserving: the marginal distribution of the feature, the distributions of all other features, the model itself (no retraining needed). This makes it a clean, controlled experiment that isolates each feature's contribution.</span>

---

## <span style="font-size: 16px;">Model-Agnostic</span>

<span style="font-size: 14px;">Unlike Gini importance (which is tree-specific) or coefficient magnitude (which is linear-model-specific), permutation importance works with any model. It only requires the ability to call the model's predict function. This makes it one of the most versatile interpretability tools.</span>

---

## <span style="font-size: 16px;">Train vs. Test Importance</span>

- <span style="font-size: 14px;">**Training set importance**: measures what the model has learned to depend on. Can be high for overfit features</span>
- <span style="font-size: 14px;">**Test set importance**: measures what actually helps generalization. More reliable for feature selection</span>

<span style="font-size: 14px;">In practice, computing importance on a held-out test set is preferred because it accounts for overfitting.</span>

---

## <span style="font-size: 16px;">Limitations</span>

- <span style="font-size: 14px;">**Correlated features**: if two features are correlated, permuting one may not drop accuracy because the other carries similar information. Both features appear less important than they are</span>
- <span style="font-size: 14px;">**Unrealistic permutations**: shuffling a feature creates data points that may not exist in the real distribution (e.g., height=2m with weight=30kg)</span>
- <span style="font-size: 14px;">**Computational cost**:</span> $O(d \cdot R \cdot n)$ <span style="font-size: 14px;">model evaluations</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

- <span style="font-size: 14px;">**Q: How do you handle correlated features?**</span>
  <span style="font-size: 14px;">A: Use conditional permutation (permute within groups), or use methods like SHAP values that properly handle feature interactions.</span>

- <span style="font-size: 14px;">**Q: Can importance be negative?**</span>
  <span style="font-size: 14px;">A: Yes, if the feature is noise and permuting it accidentally improves predictions. This typically means the feature is not useful.</span>

- <span style="font-size: 14px;">**Q: How does this compare to SHAP?**</span>
  <span style="font-size: 14px;">A: Permutation importance gives a global ranking of features. SHAP provides per-sample feature contributions and handles interactions better, but is more expensive to compute.</span>

---
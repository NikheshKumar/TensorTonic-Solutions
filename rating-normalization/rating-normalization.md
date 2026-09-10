## What Is Rating Normalization?

Rating normalization transforms raw ratings to account for systematic biases in how users rate items. Different users have different rating behaviors: some rate generously (mostly 4-5 stars), others harshly (mostly 2-3 stars). Normalization makes ratings comparable across users.

The most common normalization is mean-centering:

$$
\tilde{r}_{ui} = r_{ui} - \bar{r}_u
$$

This converts ratings to deviations from each user's personal average.

---

## Why Normalize Ratings?

**Different user scales:**

- Generous user: Likes something = 5 stars, Dislikes = 3 stars
- Harsh user: Likes something = 3 stars, Dislikes = 1 star

A "3" from the generous user means dislike; a "3" from the harsh user means like.

**Improved similarity computation:**

Without normalization, generous users appear similar to each other even if they have different preferences.

**Better predictions:**

Normalized ratings focus on preferences rather than personal rating scale.

---

## Mean-Centering (User-Based)

Subtract each user's mean rating:

$$
\tilde{r}_{ui} = r_{ui} - \bar{r}_u
$$

where $\bar{r}_u = \frac{1}{|I_u|} \sum_{i \in I_u} r_{ui}$

**Interpretation:**

- $\tilde{r}_{ui} > 0$: User rates this item above their average (they like it)
- $\tilde{r}_{ui} < 0$: User rates this item below their average (they dislike it)
- $\tilde{r}_{ui} = 0$: User rates this item at their average

---

## Worked Example: Mean-Centering

**Raw ratings:**

- User A: [5, 4, 5, 3, 5] → Mean = 4.4
- User B: [3, 2, 3, 1, 2] → Mean = 2.2

**Normalized ratings:**

- User A: [0.6, -0.4, 0.6, -1.4, 0.6]
- User B: [0.8, -0.2, 0.8, -1.2, -0.2]

**Observation:**

Both users have similar preferences (like items 1, 3, 5; dislike items 2, 4) but different raw scales. Normalization reveals this.

---

## Z-Score Normalization

Also account for rating variance:

$$
\tilde{r}_{ui} = \frac{r_{ui} - \bar{r}_u}{\sigma_u}
$$

where $\sigma_u$ is the standard deviation of user $u$'s ratings.

**Interpretation:**

- $\tilde{r}_{ui} = 1$: One standard deviation above user's mean
- $\tilde{r}_{ui} = -2$: Two standard deviations below user's mean

This handles both scale (mean) and spread (variance) differences.

---

## Z-Score Example

**User A:** Ratings [5, 4, 5, 3, 5]

- Mean = 4.4
- Std = 0.8

**User B:** Ratings [5, 3, 5, 1, 5]

- Mean = 3.8
- Std = 1.79

**Normalized rating for item with raw rating 5:**

- User A: (5 - 4.4) / 0.8 = 0.75
- User B: (5 - 3.8) / 1.79 = 0.67

Even though both gave 5 stars, User A's 5 is more exceptional relative to their pattern.

---

## Item-Based Normalization

Subtract each item's mean rating:

$$
\tilde{r}_{ui} = r_{ui} - \bar{r}_i
$$

where $\bar{r}_i = \frac{1}{|U_i|} \sum_{u \in U_i} r_{ui}$

**Use case:** When items have different quality levels, and you want to measure how much a user deviates from the consensus on each item.

---

## Combined Normalization

Account for both user and item biases:

$$
\tilde{r}_{ui} = r_{ui} - \bar{r}_u - \bar{r}_i + \mu
$$

or equivalently:

$$
\tilde{r}_{ui} = r_{ui} - (\mu + b_u + b_i)
$$

This is the residual after removing baseline effects.

---

## Normalization in User-Based CF

In user-based collaborative filtering prediction:

**Without normalization:**

$$
\hat{r}_{ui} = \frac{\sum_{v \in N(u)} \text{sim}(u, v) \cdot r_{vi}}{\sum_{v \in N(u)} |\text{sim}(u, v)|}
$$

**With normalization:**

$$
\hat{r}_{ui} = \bar{r}_u + \frac{\sum_{v \in N(u)} \text{sim}(u, v) \cdot (r_{vi} - \bar{r}_v)}{\sum_{v \in N(u)} |\text{sim}(u, v)|}
$$

Predict user's mean plus how neighbors deviate from their means.

---

## Normalization in Item-Based CF

**With normalization:**

$$
\hat{r}_{ui} = \bar{r}_i + \frac{\sum_{j \in N(i)} \text{sim}(i, j) \cdot (r_{uj} - \bar{r}_j)}{\sum_{j \in N(i)} |\text{sim}(i, j)|}
$$

Or using user normalization:

$$
\hat{r}_{ui} = \bar{r}_u + \frac{\sum_{j \in N(i)} \text{sim}(i, j) \cdot (r_{uj} - \bar{r}_u)}{\sum_{j \in N(i)} |\text{sim}(i, j)|}
$$

---

## De-Normalization

After predicting normalized ratings, convert back to original scale:

**For mean-centering:**

$$
\hat{r}_{ui} = \tilde{r}_{ui} + \bar{r}_u
$$

**For Z-score:**

$$
\hat{r}_{ui} = \tilde{r}_{ui} \cdot \sigma_u + \bar{r}_u
$$

Always de-normalize before presenting predictions to users.

---

## Effect on Similarity

**Pearson correlation = Cosine on mean-centered data:**

$$
\text{Pearson}(u, v) = \text{Cosine}(\tilde{\mathbf{r}}_u, \tilde{\mathbf{r}}_v)
$$

Mean-centering removes user bias before computing similarity.

**Adjusted cosine = Pearson with user-centering for items:**

Used in item-based CF to account for user rating scales.

---

## Handling Edge Cases

**Single rating:**

If user has only one rating, mean = that rating, variance = 0.

- Mean-centering: normalized rating = 0
- Z-score: undefined (division by zero)

**Solution:** Use global mean or minimum variance threshold.

**Identical ratings:**

If all ratings are the same (e.g., all 5s), variance = 0.

**Solution:** Skip normalization or use fallback.

---

## Min-Max Normalization

Transform ratings to [0, 1] range:

$$
\tilde{r}_{ui} = \frac{r_{ui} - r_{min}}{r_{max} - r_{min}}
$$

where $r_{min}$ and $r_{max}$ are the rating scale bounds (e.g., 1 and 5).

This does not account for user-specific biases.

---

## Decimal Scaling

For ratings with large variance:

$$
\tilde{r}_{ui} = \frac{r_{ui}}{10^k}
$$

where $k$ is chosen so $|\tilde{r}_{ui}| < 1$.

Less common in recommendations but used in some contexts.

---

## When to Normalize

**Recommended:**

- Computing user-user or item-item similarity
- Using neighborhood-based collaborative filtering
- When users have visibly different rating patterns

**Optional:**

- Matrix factorization (can learn biases explicitly)
- Deep learning models (can learn normalization)

**Not needed:**

- Binary implicit feedback (no scale differences)
- Rank-based evaluation (only order matters)

---

## Visualization of Normalization Effect

**Before normalization:**

Users with high means cluster together, low means cluster together, regardless of actual preferences.

**After normalization:**

Users with similar taste patterns cluster together, regardless of their rating scale.

---

## Normalization and Sparsity

Normalization uses statistics from observed ratings only:

$$
\bar{r}_u = \frac{1}{|I_u|} \sum_{i \in I_u} r_{ui}
$$

Users with few ratings have less reliable means. Consider:

- Shrinkage toward global mean
- Minimum rating count threshold
- Bayesian estimation of user mean
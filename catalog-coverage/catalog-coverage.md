## What Is Catalog Coverage?

Catalog coverage is a metric that measures what fraction of the item catalog a recommender system actually recommends to users. It quantifies the diversity of recommendations across the entire item space.

$$
\text{Catalog Coverage} = \frac{|\text{Recommended Items}|}{|\text{Total Items}|}
$$

A system with high catalog coverage recommends many different items; one with low coverage concentrates recommendations on a small subset.

---

## Why Catalog Coverage Matters

**Long tail problem:**

Most recommender systems exhibit popularity bias, recommending popular items far more often than niche items. This leaves the "long tail" of less popular items largely invisible.

**Business value:**

Unrecommended items generate no sales from recommendations. If 90% of your catalog is never recommended, you're missing opportunities.

**User satisfaction:**

Users may have diverse tastes. Showing only popular items misses users who prefer niche content.

**Content providers:**

In platforms with content creators (YouTube, Spotify), low coverage means most creators get no exposure from recommendations.

---

## Computing Catalog Coverage

**Step 1:** Generate recommendations for all users (or a sample)

**Step 2:** Collect the union of all recommended items

**Step 3:** Divide by total catalog size

$$
\text{Coverage} = \frac{|\bigcup_{u} R_u|}{|I|}
$$

where $R_u$ is the set of items recommended to user $u$ and $I$ is the full item catalog.

---

## Worked Example

**Catalog:** 100 items total

**Recommendations:**

- User 1: Items {1, 2, 3, 5, 10}
- User 2: Items {1, 2, 4, 6, 10}
- User 3: Items {1, 3, 5, 7, 10}
- User 4: Items {2, 3, 4, 8, 10}

**Union of recommended items:**

{1, 2, 3, 4, 5, 6, 7, 8, 10} = 9 unique items

**Catalog coverage:**

$$
\text{Coverage} = \frac{9}{100} = 0.09 = 9\%
$$

Only 9% of the catalog ever gets recommended.

---

## Coverage at Different K Values

Coverage depends on how many items you recommend per user:

**Coverage@5:** Coverage when recommending top-5 items to each user

**Coverage@10:** Coverage when recommending top-10 items

**Coverage@N:** Coverage when recommending top-N items

Larger K generally increases coverage, but the relationship is not linear due to overlap.

---

## Coverage Over Time

In dynamic systems, measure coverage over a time window:

**Daily coverage:** What fraction of items were recommended today?

**Weekly coverage:** Over a week, what fraction of items appeared in recommendations?

**Cumulative coverage:** What fraction has ever been recommended?

Time-based coverage captures whether the system explores the catalog over time or stays stuck on the same items.

---

## Relationship to Recommendation Frequency

Let $f_i$ be the number of users to whom item $i$ was recommended:

$$
\text{Coverage} = \frac{|\{i : f_i > 0\}|}{|I|}
$$

Coverage only cares whether an item was recommended at least once, not how often.

**Gini coefficient** and **entropy** capture the distribution of recommendation frequency, complementing coverage.

---

## Coverage vs Accuracy

There is often a tradeoff between coverage and accuracy:

**High accuracy, low coverage:**

Recommend only items the system is confident about. These tend to be popular items with many ratings.

**High coverage, lower accuracy:**

Force the system to recommend diverse items, including less certain predictions.

**The balance:**

Business goals determine the right tradeoff. Pure accuracy optimization often sacrifices coverage.

---

## Popularity Bias and Coverage

Popularity-based recommenders have inherently low coverage:

**If you always recommend the top-100 popular items:**

Coverage = 100 / |I|

For a catalog of 10,000 items, that is only 1% coverage.

**Collaborative filtering** can have better coverage by finding niche items that match specific user preferences.

---

## Improving Coverage

**Diversity injection:**

After generating top-N candidates, replace some with diverse alternatives.

**Exploration bonuses:**

Boost scores for rarely-recommended items.

**Multi-objective optimization:**

Optimize for both relevance and coverage jointly.

**Calibrated recommendations:**

Match the distribution of recommended items to user preferences across genres/categories.

---

## Coverage by Category

Instead of overall coverage, measure coverage within categories:

$$
\text{Coverage}_{category} = \frac{|\text{Recommended items in category}|}{|\text{Items in category}|}
$$

This reveals whether certain categories are underserved.

**Example:**

- Action movies: 80% coverage
- Documentary films: 5% coverage

The system heavily favors action movies.

---

## Coverage vs Personalization

**High coverage, low personalization:**

Every user gets different random items. Coverage is high but recommendations are not useful.

**High coverage, high personalization:**

Different users get recommendations tailored to their tastes, collectively covering much of the catalog.

**Low coverage, high personalization:**

All users with similar tastes get the same recommendations. Good for individuals but collectively narrow.

The goal is high coverage WITH good personalization.

---

## Aggregate vs Individual Coverage

**Aggregate coverage (discussed above):**

What fraction of items appear in ANY user's recommendations?

**Individual coverage:**

For a single user, what fraction of items they might like are recommended to them?

Both matter. Aggregate coverage measures system-level diversity; individual coverage measures whether users discover their potential favorites.

---

## Coverage in Different Domains

**E-commerce:**

Low coverage means unsold inventory. Recommendations should help move long-tail products.

**Streaming (Netflix, Spotify):**

Low coverage means content creators get no exposure. Platform health requires discovering new content.

**News:**

Low coverage means only viral stories are shown. Diverse coverage supports informed citizens.

**Job recommendations:**

Low coverage means many jobs never get applicants from recommendations.

---

## Measuring Coverage in Practice

**Sample-based estimation:**

For large systems, generate recommendations for a random sample of users, then extrapolate.

**Production logging:**

Track which items appear in recommendations served to real users.

**A/B testing:**

Compare coverage between different recommendation algorithms.

---

## Coverage and the Filter Bubble

Low coverage contributes to filter bubbles:

- Users only see a narrow slice of content
- Similar users see similar narrow slices
- Diverse perspectives and niche content become invisible

Higher coverage can help burst filter bubbles by exposing users to broader content.

---

## Theoretical Bounds

**Minimum coverage:**

If you recommend k items to n users, minimum coverage occurs when all users get the same k items.

$$
\text{Coverage}_{min} = \frac{k}{|I|}
$$

**Maximum coverage:**

Maximum coverage occurs when there is no overlap.

$$
\text{Coverage}_{max} = \min\left(1, \frac{n \cdot k}{|I|}\right)
$$

Actual coverage falls between these bounds.
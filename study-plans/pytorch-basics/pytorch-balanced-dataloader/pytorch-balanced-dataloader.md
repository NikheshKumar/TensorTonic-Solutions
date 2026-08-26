## <span style="font-size: 20px;">The Class Imbalance Problem</span>

- Most real-world datasets have skewed class distributions: fraud detection (99.9% legitimate), medical imaging (thousands of normals per rare disease), spam filtering (5% to 50% spam)
- A model that always predicts the majority class achieves high accuracy but is completely useless for detecting the minority class
- Gradient-based training with uniform sampling produces gradients dominated by the majority class. Over many batches, the model learns that minimizing loss on the majority class is the most efficient path to low average loss, neglecting the minority class
- The decision boundary shifts toward the minority class because the majority class exerts more influence on parameter updates
- In learned representations, minority-class samples often cluster near the decision boundary or scatter without clear structure, while majority-class samples form tight, well-separated clusters

**Example:** 950 samples of class 0 and 50 of class 1. A model always predicting class 0 achieves 95% accuracy but has zero recall, zero precision on the positive class, and an F1 of 0.

## Why Accuracy is Misleading

$$
\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}
$$

- When class sizes are imbalanced, the denominator is dominated by the majority class, giving disproportionate credit for correct majority-class predictions
- More informative metrics for imbalanced settings:
  - **Precision:** $\frac{\text{TP}}{\text{TP} + \text{FP}}$ - of all samples predicted positive, what fraction are actually positive. High precision means few false alarms
  - **Recall (sensitivity):** $\frac{\text{TP}}{\text{TP} + \text{FN}}$ - of all actually positive samples, what fraction are correctly identified. High recall means few missed positives
  - **F1 score:** $\frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ - harmonic mean, penalizes models that sacrifice one for the other
  - **AUROC:** measures ability to rank positives above negatives across all classification thresholds. 0.5 = random guessing, 1.0 = perfect separation. Threshold-independent, useful for model comparison
  - **AUPRC:** especially informative when the positive class is very rare. Unlike AUROC, which can appear high even with poor precision, AUPRC directly measures performance in the relevant operating region

## Sampling Strategies for Imbalanced Data

## Oversampling

- Draw minority-class samples more frequently than their natural proportion (sampled with replacement)
- In the 950/50 example, minority samples are drawn roughly 19 times each per epoch, so both classes contribute approximately 950 samples
- Risk: overfitting to minority class since the same samples are seen many times. The model may memorize them rather than learning generalizable features
- Mitigation: apply data augmentation to oversampled minority instances to present each repeated sample in a slightly different form

## Undersampling

- Reduce majority-class samples to match minority class size
- In the 950/50 example, randomly select 50 from class 0, yielding a balanced dataset of 100
- Advantage: training is much faster (100 samples vs 1900)
- Disadvantage: you discard 900 majority-class samples, potentially losing important distribution information
- Can be effective when the majority class is dense (many samples near the same decision boundary)

## SMOTE (Synthetic Minority Over-sampling Technique)

Generates new minority-class samples by interpolating between existing ones:

$$
\mathbf{x}_{\text{new}} = \mathbf{x} + \lambda \cdot (\mathbf{x}' - \mathbf{x})
$$

- $\mathbf{x}$ is a minority sample, $\mathbf{x}'$ is one of its $k$-nearest minority-class neighbors, $\lambda \sim \text{Uniform}(0, 1)$
- Produces new points along the line segment between existing samples, enriching the feature space rather than repeating data
- Most effective in low-dimensional feature spaces. In high-dimensional spaces (images, text embeddings), interpolation can produce unrealistic samples outside the data manifold
- Typically applied as a preprocessing step before training, not inside the DataLoader

## Weighted Random Sampling: The Mathematics

Suppose $C$ classes, class $c$ contains $n_c$ samples, total $N = \sum_{c=1}^{C} n_c$. Assign a sampling weight $w_i$ to each sample $i$ such that drawing $N$ samples according to these weights gives equal class representation in expectation.

The probability of drawing sample $i$ in a single draw:

$$
P(i) = \frac{w_i}{\sum_{j=1}^{N} w_j}
$$

The expected number of times class $c$ is drawn in $N$ independent draws:

$$
\mathbb{E}[\text{count of class } c] = N \cdot \frac{n_c \cdot w_c}{\sum_{k=1}^{C} n_k \cdot w_k}
$$

where $w_c$ is the weight assigned to every sample in class $c$. For uniform class representation, we want this to be the same for all classes. Setting $w_c = 1 / n_c$ satisfies this:

$$
n_c \cdot w_c = n_c \cdot \frac{1}{n_c} = 1 \quad \Rightarrow \quad \sum_k n_k \cdot w_k = C
$$

$$
\frac{n_c \cdot w_c}{\sum_k n_k \cdot w_k} = \frac{1}{C} \quad \checkmark
$$

Each class contributes $N / C$ samples in expectation per epoch. For the 950/50 binary example, $w_0 = 1/950 \approx 0.00105$ and $w_1 = 1/50 = 0.02$. Class 1 samples are about 19 times more likely to be drawn, exactly compensating for their 19:1 underrepresentation.

## Alternative Weight Schemes

**Temperature-scaled balancing:**

$$
w_c = \frac{1}{n_c^{\alpha}}
$$

where $\alpha \in [0, 1]$ controls rebalancing strength. At $\alpha = 0$: all weights are 1 (uniform sampling). At $\alpha = 1$: full inverse-frequency balancing. Values in between provide smooth interpolation. Useful when the minority class is very small and full balancing would cause extreme oversampling.

**Effective number of samples** (Cui et al. 2019): as you see more samples from a class, each additional sample provides diminishing information due to overlap in feature space. The effective number:

$$
E_n = \frac{1 - \beta^n}{1 - \beta}
$$

where $\beta \in [0, 1)$. Weight: $w_c = 1 / E_{n_c}$. At $\beta = 0$: equivalent to inverse-frequency. At $\beta \to 1$: equivalent to uniform sampling.

## WeightedRandomSampler in PyTorch

The WeightedRandomSampler class takes three parameters:

- **weights:** a sequence of $N$ non-negative floats, one per sample in the dataset. These are the sampling weights $w_i$. They do not need to sum to 1; PyTorch normalizes them internally
- **num_samples:** the number of indices to draw per epoch. Setting this to $N$ keeps the epoch length the same as standard training. You can set it to a different value for shorter or longer epochs
- **replacement:** a boolean. When True, the same index can be drawn multiple times in a single epoch. When oversampling minority classes, replacement must be True, because you need to draw minority samples more times than they exist in the dataset. When False, each index appears at most once, preventing full oversampling

Internally, the sampler uses multinomial sampling to draw indices according to the weight distribution. Each call to the iterator produces a fresh set of num_samples indices, where the probability of drawing index $i$ is proportional to $w_i$.

Construction pattern:

- Compute class counts from labels (e.g., via bincount)
- Compute class weights as 1.0 / class_counts (as float tensors)
- Broadcast to per-sample weights by indexing class weights with labels: if labels = [0, 0, 1, 0, 1] and class_weights = [0.333, 1.0], then sample_weights = [0.333, 0.333, 1.0, 0.333, 1.0]. This broadcasting is the key step that converts per-class weights into per-sample weights
- Create the sampler with these weights, num_samples = len(labels), replacement = True
- Pass to DataLoader via the sampler parameter (cannot use shuffle=True with a custom sampler, since the sampler controls ordering)

## Replacement Sampling: Implications

With replacement, the sampler draws indices independently. In $N$ draws from $N$ items:

$$
\mathbb{E}[\text{unique samples}] = N \left(1 - \left(1 - \frac{1}{N}\right)^N\right) \approx 0.632N
$$

- About 36.8% of samples are missed in any given epoch, even with uniform weights
- With non-uniform weights, this effect is more pronounced for majority-class samples (lower weights) and less for minority-class samples (higher weights, drawn frequently)
- This is a feature, not a bug: the goal is balanced class distribution, not seeing every sample once
- Over multiple epochs, every sample is eventually seen
- The stochastic variation acts as a mild regularizer, similar to dropout
- Without replacement: each index appears at most once. Full oversampling is impossible, but useful for stratified sampling (proportional rather than equal representation)

## Per-Sample vs. Per-Class Weights

- The sampler operates with per-sample weights, but typically all samples in the same class receive the same weight
- More general per-sample weighting can be based on:
  - **Sample difficulty:** upweight samples the model gets wrong more often (curriculum learning, hard example mining)
  - **Data quality:** downweight noisy or mislabeled samples using a confidence score
  - **Recency:** upweight recently added samples to adapt to distribution shifts
- The conversion from class weights to sample weights is a simple indexing operation

## Class-Weighted Loss: Alternative Approach

Instead of modifying the sampling distribution, modify the loss function. For a sample from class $c$ with model output $\hat{y}$:

$$
\mathcal{L}_{\text{weighted}}(\hat{y}, c) = -w_c \log \hat{y}_c
$$

- Mathematically related to sampling-based balancing: inverse-frequency weights in the loss produce approximately the same expected gradient contribution per class over a uniformly sampled epoch
- However, the equivalence is only approximate. Weighted loss scales gradient magnitude per sample, while weighted sampling changes which samples the model sees. They differ in interaction with batch normalization, momentum optimizers, and regularization
- Both work well in practice and can be combined for additional effect

## Focal Loss

Weights by prediction confidence rather than class frequency:

$$
\mathcal{L}_{\text{focal}}(\hat{y}, c) = -(1 - \hat{y}_c)^\gamma \log \hat{y}_c
$$

- When the model is confident ($\hat{y}_c$ near 1): the factor $(1 - \hat{y}_c)^\gamma$ is near zero, loss contribution is small
- When uncertain ($\hat{y}_c$ near 0): the factor is near 1, loss contribution is full-strength
- Automatically downweights easy examples (mostly majority class) and upweights hard examples (often minority or ambiguous)
- Can combine with class weights: $\mathcal{L} = -\alpha_c (1 - \hat{y}_c)^\gamma \log \hat{y}_c$

## Relationship Between Sampling and Loss Weighting

Let $\ell_i$ be the unweighted loss and $\nabla \ell_i$ the gradient for sample $i$.

Under uniform sampling with weighted loss (weight $w_c$):

$$
\mathbb{E}[\nabla \mathcal{L}_{\text{weighted}}] = \sum_{c=1}^{C} \sum_{i \in \text{class } c} w_c \nabla \ell_i
$$

Under weighted sampling with unweighted loss ($w_c = 1/n_c$, $N$ draws with replacement):

$$
\begin{aligned}
\mathbb{E}[\nabla \mathcal{L}_{\text{sampled}}]
&\propto \sum_{c=1}^{C} \frac{1}{n_c} \sum_{i \in \text{class } c} \nabla \ell_i \\
&= \sum_{c=1}^{C} \overline{\nabla \ell}_c
\end{aligned}
$$

The key difference: weighted loss scales each sample's gradient individually, while weighted sampling changes how often each sample appears. For single-sample SGD, expected gradients are proportional. For minibatch SGD, batch composition affects gradient variance, which influences training dynamics.

## Stratified Sampling

Stratified sampling is a related but distinct concept from balanced sampling:

- Balanced sampling aims for equal class representation in each epoch
- Stratified sampling aims to preserve the natural class proportions in each batch
- In a dataset with 95% class 0 and 5% class 1, stratified sampling ensures each batch has approximately 95%/5% split, rather than the random fluctuations from uniform sampling
- Most useful for evaluation (stable metric estimates) and train/validation splitting (preserving class distribution)
- Stratified sampling alone does not solve class imbalance: it preserves the imbalance. But it guarantees every batch contains at least some minority-class samples, stabilizing gradient estimates
- PyTorch does not provide a built-in stratified sampler, but you can implement one by maintaining separate index lists per class and interleaving according to desired proportions

## Practical Considerations

## When to Use Balanced Sampling

- Moderate to severe imbalance (less than 1:10 ratio)
- The minority class is the class you care most about (fraud, disease detection)
- Dataset is large enough that oversampled minority instances will not be memorized easily

## When Not to Use It

- If the imbalance reflects the true deployment distribution and you need calibrated probabilities
- Balanced sampling distorts probability estimates: the model expects equal proportions but encounters skewed ones in production
- Alternative: train with natural distribution and recalibrate outputs using Platt scaling or temperature scaling

## Monitoring Class Balance During Training

After setting up balanced sampling, verify it is working by counting the class distribution in a few batches. Iterate over one epoch and tally the labels. With properly configured inverse-frequency weights and replacement sampling, the class counts should be approximately equal, subject to sampling noise.

## Interaction with Batch Normalization

- Batch norm computes statistics from the current batch
- With balanced sampling, batch statistics reflect the balanced distribution, not the natural one
- This can cause a mismatch at inference time, when running statistics (accumulated during training) are used
- Usually minor for moderate imbalances; for extreme cases, consider group or layer normalization

## Multi-Label and Multi-Class Scenarios

- The inverse-frequency weighting scheme extends naturally to multi-class problems: compute $w_c = 1/n_c$ for each of $C$ classes
- For multi-label problems (each sample can belong to multiple classes), the concept of "class count" is less clear, and sampling-based balancing is harder to apply
- In multi-label cases, loss weighting (using binary cross-entropy with a pos_weight parameter) is typically more appropriate

## Evaluation Metrics: Deeper Look

- **Confusion matrix:** $C \times C$ matrix where entry $(i, j)$ is the count of true class $i$ predicted as class $j$. Diagonal = correct, off-diagonal = errors. Makes systematic misclassification patterns immediately visible
- **Macro-averaged metrics:** compute per class, take unweighted mean. Gives equal importance to every class regardless of size
- **Micro-averaged metrics:** aggregate contributions of all classes, then compute. Micro precision = accuracy for multi-class
- **Weighted-averaged metrics:** weight each class's metric by its support, providing a compromise between macro and micro
- **Cohen's kappa:** measures agreement corrected for chance. 0 = no better than random, 1 = perfect
- **Matthews Correlation Coefficient (MCC):** accounts for all four confusion matrix cells, value in $[-1, +1]$, symmetric with respect to positive and negative classes. Considered one of the most balanced metrics for binary classification

## Combining Approaches

The best results on severely imbalanced datasets come from combining multiple techniques:

- Use balanced sampling or moderate oversampling to ensure the model sees minority-class samples frequently
- Apply data augmentation to oversampled minority instances to reduce memorization risk
- Use a class-weighted loss or focal loss to further emphasize hard or minority-class samples in the gradient computation
- Evaluate using class-balanced metrics (macro F1, AUPRC) rather than accuracy
- Tune the balance hyperparameters ($\alpha$ in the weight scheme, $\gamma$ in focal loss, the oversampling ratio) using a validation set with the natural class distribution
- The validation set should always reflect the natural distribution, even when the training set is rebalanced. This ensures that hyperparameter choices optimize real-world performance, not performance on an artificially balanced distribution

## Summary

Class imbalance is a pervasive problem that degrades model performance on minority classes unless explicitly addressed. Accuracy is misleading in imbalanced settings; precision, recall, F1, AUROC, and AUPRC provide more informative evaluations. Weighted random sampling addresses imbalance at the data loading level by assigning each sample a weight inversely proportional to its class frequency, $w_c = 1/n_c$, ensuring every class contributes equally in expectation. The sampler draws indices according to the weight distribution, with replacement, producing balanced batches from an unbalanced dataset. Alternative approaches include class-weighted loss functions and focal loss, which modify the gradient contribution rather than the sampling distribution. The most robust approach combines balanced sampling, augmentation, and appropriate loss weighting, validated on a naturally distributed holdout set.

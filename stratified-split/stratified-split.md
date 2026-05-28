## What is Stratified Splitting?

Stratified splitting divides a dataset into subsets while preserving the proportion of each class. If the original dataset has 70% class A and 30% class B, both the training and test sets will maintain approximately these same proportions. This is crucial for imbalanced classification problems.

---

## Why Stratify?

**Preserves class distribution**: Random splitting can accidentally put most minority class samples in one split, leaving the other split unrepresentative.

**Reliable evaluation**: Test set performance reflects real-world distribution when class proportions are maintained.

**Consistent training signal**: Training set contains sufficient examples of all classes.

**Essential for imbalanced data**: When one class is rare (e.g., 1% fraud cases), random splitting could result in a test set with zero fraud cases.

---

## The Problem with Random Splitting

**Example scenario**:
- 1000 samples: 950 class A, 50 class B (5% minority)
- 80/20 random split

**Possible random outcome**:
- Training: 760 class A, 40 class B (5.0%)
- Test: 190 class A, 10 class B (5.0%)

This is lucky. But random splits can also produce:
- Training: 770 class A, 30 class B (3.7%)
- Test: 180 class A, 20 class B (10.0%)

The test set has double the minority class proportion - evaluation will be misleading.

---

## Stratified Splitting Process

**Step 1**: Group samples by class label

**Step 2**: For each class, randomly split into train/test with the specified ratio

**Step 3**: Combine all class-specific train samples into the final training set

**Step 4**: Combine all class-specific test samples into the final test set

**Result**: Both sets have the same class proportions as the original data.

---

## Mathematical Formulation

For a dataset with $N$ samples and class distribution:
- Class 0: $n_0$ samples (proportion $p_0 = n_0/N$)
- Class 1: $n_1$ samples (proportion $p_1 = n_1/N$)
- ...

With train fraction $f$ (e.g., 0.8 for 80% train):

**Training set**:
- Class 0: $\lfloor f \cdot n_0 \rfloor$ samples
- Class 1: $\lfloor f \cdot n_1 \rfloor$ samples
- ...

**Test set**:
- Class 0: $n_0 - \lfloor f \cdot n_0 \rfloor$ samples
- Class 1: $n_1 - \lfloor f \cdot n_1 \rfloor$ samples
- ...

---

## Worked Example

**Dataset**: 100 samples
- Class A: 70 samples (70%)
- Class B: 20 samples (20%)
- Class C: 10 samples (10%)

**Split ratio**: 80% train, 20% test

**Stratified split calculation**:

Class A:
- Train: $\lfloor 0.8 \times 70 \rfloor = 56$ samples
- Test: $70 - 56 = 14$ samples

Class B:
- Train: $\lfloor 0.8 \times 20 \rfloor = 16$ samples
- Test: $20 - 16 = 4$ samples

Class C:
- Train: $\lfloor 0.8 \times 10 \rfloor = 8$ samples
- Test: $10 - 8 = 2$ samples

**Training set**: 56 + 16 + 8 = 80 samples
- Class A: 56/80 = 70%
- Class B: 16/80 = 20%
- Class C: 8/80 = 10%

**Test set**: 14 + 4 + 2 = 20 samples
- Class A: 14/20 = 70%
- Class B: 4/20 = 20%
- Class C: 2/20 = 10%

**Both sets preserve the original 70/20/10 distribution.**

---

## Handling Small Classes

When a class has very few samples, stratified splitting faces challenges:

**Example**: Class with 3 samples, 80/20 split
- Train: $\lfloor 0.8 \times 3 \rfloor = 2$ samples
- Test: 1 sample

This works, but with only 2 samples, one split may have 2 or 0.

**Solutions**:
- Ensure minimum samples per class before splitting
- Use leave-one-out for tiny classes
- Consider combining rare classes

---

## Stratified K-Fold Cross-Validation

Extends stratification to K-Fold:

**Process**:
1. Group samples by class
2. Within each class, divide into K folds
3. Each fold contains proportional representation of all classes
4. Iterate: use each fold as validation, others as training

**Benefit**: Every fold has representative class distribution for reliable cross-validation estimates.

---

## Multi-Label Stratification

When samples can have multiple labels (multi-label classification):

**Challenge**: Simple stratification on single labels does not work

**Iterative stratification algorithm**:
1. Order labels by frequency (rarest first)
2. For each sample with the rarest label, assign to the fold with smallest proportion of that label
3. Repeat for next rarest label

**Goal**: Approximately preserve the distribution of all label combinations

---

## Implementation Considerations

**Shuffling within classes**: Randomly shuffle samples within each class before splitting to avoid ordering bias

**Reproducibility**: Set random seed for consistent splits across runs

**Rounding**: Floor function ensures train set gets the integer count; remaining go to test

**Extremely rare classes**: May need special handling or minimum sample requirements

---

## Stratification vs Random Split

**Use stratified split when**:
- Class distribution is imbalanced
- Evaluation metric depends on class proportions
- Training requires examples of all classes
- Dataset is small (random variation is higher)

**Random split acceptable when**:
- Classes are roughly balanced
- Dataset is very large (random variation is minimal)
- Task is regression (no classes to stratify)

---

## Regression Stratification

For regression tasks, stratify on binned target values:

**Process**:
1. Bin continuous target into discrete intervals
2. Treat bins as pseudo-classes
3. Apply stratified splitting on bins
4. Ensures both splits have similar target distributions

**Example**: Income prediction
- Bin incomes into [0-30k], [30k-60k], [60k-100k], [100k+]
- Stratify based on these bins

---

## Where Stratified Splitting Shows Up

- **Medical Diagnosis**: Rare diseases require stratification to ensure test sets contain positive cases

- **Fraud Detection**: Fraud cases are rare; stratification ensures both splits see fraud examples

- **Sentiment Analysis**: When positive/negative/neutral are unbalanced

- **Object Detection**: Some object classes appear more frequently than others

- **Customer Churn**: Churned customers are typically a small fraction

- **Credit Scoring**: Defaults are rare events requiring careful split handling

- **A/B Testing Analysis**: Stratify on user segments for representative results

- **Model Selection**: Cross-validation requires stratification for reliable hyperparameter tuning

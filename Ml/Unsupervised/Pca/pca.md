# PCA — Interview Questions & Answers With Examples

## 1. What is PCA?

**Answer:**

> PCA (Principal Component Analysis) is an unsupervised dimensionality-reduction technique. It transforms the original features into new features called principal components and keeps the components that capture the most variance.

### Example

Suppose our customer dataset has:

```text
Age
Annual_Income
Spending_Score
```

PCA can transform:

```text
3 original features
        ↓
       PCA
        ↓
PC1
PC2
PC3
```

If we keep only PC1 and PC2:

```text
3 features → 2 components
```

---

# 2. Why do we use PCA?

**Answer:**

> We use PCA when we have many features and want to reduce the number of dimensions while preserving most of the important variation in the data.

### Example

```text
100 features
     ↓
    PCA
     ↓
20 components
```

Instead of processing 100 features, we process 20 components.

---

# 3. What is dimensionality reduction?

**Answer:**

> Dimensionality reduction means reducing the number of features/dimensions while preserving important information.

### Example

```text
Age
Income
Spending
Visits
Purchases
Reviews
...
100 features
     ↓ PCA
20 components
```

---

# 4. Does PCA remove columns?

**Answer:**

> No. PCA does not simply remove columns. It creates new features called principal components from combinations of the original features.

### Example

Original:

```text
Age
Income
Spending
```

PCA creates:

```text
PC1
PC2
PC3
```

PC1 might conceptually be:

```text
PC1 =
0.5 × Age
+ 0.7 × Income
- 0.4 × Spending
```

So PC1 is a **new feature**, not an existing column.

---

# 5. What is a Principal Component?

**Answer:**

> A principal component is a new feature created as a linear combination of the original features.

### Example

```text
Age ─────────┐
Income ──────┼──→ PC1
Spending ────┘
```

For example:

```text
PC1 = 0.5 Age + 0.7 Income - 0.4 Spending
```

The actual weights are calculated by PCA.

---

# 6. What is PC1?

**Answer:**

> PC1 is the direction/component that captures the maximum variance in the dataset.

### Example

Suppose:

```text
PC1 → 70%
PC2 → 20%
PC3 → 10%
```

PC1 contains the largest amount of variance.

---

# 7. What is PC2?

**Answer:**

> PC2 captures the next highest amount of variance after PC1 and is orthogonal to PC1.

Example:

```text
PC1 → 70%
PC2 → 20%
PC3 → 10%
```

PC2 captures the second-highest amount.

---

# 8. What does variance mean in PCA?

**Answer:**

> Variance represents how much the data varies along a particular direction.

### Simple example

Suppose customers have:

```text
Customer A → Income = 30K
Customer B → Income = 31K
Customer C → Income = 32K
```

There isn't much variation.

But:

```text
Customer A → 30K
Customer B → 70K
Customer C → 150K
```

has much larger variation.

PCA searches for directions where the data varies the most.

---

# 9. Why does PCA try to maximize variance?

**Answer:**

> PCA assumes that directions with larger variance capture more of the useful structure of the dataset, so it prioritizes those directions.

Example:

```text
PC1 → 70%
PC2 → 20%
PC3 → 10%
```

PC1 is more important in terms of variance than PC3.

---

# 10. What is explained variance ratio?

**Answer:**

> Explained variance ratio tells us what proportion of the total variance is captured by each principal component.

### Example

```text
PC1 → 0.70 → 70%
PC2 → 0.20 → 20%
PC3 → 0.10 → 10%
```

Total:

```text
70 + 20 + 10 = 100%
```

---

# 11. What is cumulative explained variance?

**Answer:**

> Cumulative explained variance is the total variance captured when we progressively add principal components.

### Example

```text
PC1 → 70%
PC2 → 20%
PC3 → 10%
```

Cumulative:

```text
PC1          → 70%
PC1 + PC2    → 90%
PC1 + PC2 + PC3 → 100%
```

---

# 12. How do you decide how many components to keep?

**Answer:**

> We look at cumulative explained variance and choose enough components to retain an acceptable amount of variance for the problem.

### Example

Suppose:

```text
PC1 → 70%
PC2 → 20%
PC3 → 10%
```

If our target is approximately 90%:

```text
PC1 + PC2 = 90%
```

So we keep:

```text
2 components
```

Instead of:

```text
3 components
```

---

# 13. What happens if we keep all components?

**Answer:**

> If we keep all components, PCA has transformed the data but has not reduced its dimensionality.

### Example

```text
3 features
   ↓ PCA
3 components
```

Dimensions are still 3.

But:

```text
3 features
   ↓ PCA
2 components
```

is dimensionality reduction.

---

# 14. Why do we scale before PCA?

**Answer:**

> PCA is affected by feature scale. If features have very different scales, features with larger numerical scales can dominate the variance calculation. Therefore, we commonly standardize the features before PCA.

### Example

```text
Age             → 20–60
Annual Income   → 20,000–150,000
Spending Score  → 1–100
```

Income has a much larger numerical scale.

So we do:

```python
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

Then PCA:

```python
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)
```

---

# 15. What happens inside PCA?

**Answer:**

Conceptually:

```text
Original data
     ↓
Standardized data
     ↓
Analyze variance/correlation structure
     ↓
Find principal directions
     ↓
Find PC1
     ↓
Find PC2
     ↓
Find PC3
     ↓
Calculate explained variance
     ↓
Choose components
     ↓
Transform data
```

---

# 16. Example of PCA on our customer dataset

Original:

| Customer | Age | Income | Spending |
| -------- | --: | -----: | -------: |
| A        |  20 |     30 |       80 |
| B        |  22 |     32 |       78 |
| C        |  25 |     35 |       75 |
| D        |  50 |     90 |       20 |
| E        |  52 |     95 |       15 |

We have:

```text
3 features
```

After scaling:

```text
Age
Income
Spending
```

Then PCA might give:

```text
PC1 → 72%
PC2 → 23%
PC3 → 5%
```

If we want 95%:

```text
PC1 + PC2
= 72% + 23%
= 95%
```

Therefore:

```text
Before:

Age
Income
Spending

3 dimensions


After:

PC1
PC2

2 dimensions
```

---

# 17. What happens to Customer A after PCA?

Before:

```text
Customer A

Age = 20
Income = 30
Spending = 80
```

After PCA, the same customer might be represented as:

```text
Customer A

PC1 = -1.5
PC2 = 0.2
```

The exact values come from the PCA transformation.

The customer hasn't disappeared.

Only the **representation of the customer has changed**.

---

# 18. What happens to the original features?

**Answer:**

> The original features are transformed into principal components.

Example:

```text
Age ─────────┐
Income ──────┼──→ PC1
Spending ────┤
             └──→ PC2
```

PC1 and PC2 contain information from the original features.

---

# 19. PCA vs Feature Selection?

**Answer:**

Feature selection keeps original features.

```text
Age
Income
Spending
    ↓
Keep Age + Income
```

PCA creates new features.

```text
Age
Income
Spending
    ↓ PCA
PC1
PC2
```

So:

> **Feature selection selects existing features; PCA creates new features.**

---

# 20. Is PCA supervised or unsupervised?

**Answer:**

> PCA is generally considered an unsupervised technique because it does not require a target variable.

Example:

```text
X = Age, Income, Spending

No y required
```

---

# 21. Does PCA need `y`?

**Answer:**

No.

```python
pca.fit_transform(X_scaled)
```

Only the feature matrix is required.

---

# 22. What is `n_components`?

**Answer:**

> `n_components` specifies how many principal components we want to keep.

Example:

```python
pca = PCA(n_components=2)
```

means:

```text
3 features
   ↓
PCA
   ↓
2 components
```

---

# 23. Can `n_components` be a percentage?

**Answer:**

Yes. In scikit-learn, we can specify a float between 0 and 1 to retain that fraction of variance.

Example:

```python
pca = PCA(n_components=0.95)
```

This means:

> Keep enough components to explain at least 95% of the variance.

---

# 24. What is `fit_transform()` in PCA?

**Answer:**

It does two things:

```text
fit
 ↓
Learn principal components

transform
 ↓
Convert data into principal-component space
```

Example:

```python
X_pca = pca.fit_transform(X_scaled)
```

---

# 25. What does `fit()` do in PCA?

**Answer:**

> `fit()` learns the principal components from the training data.

```python
pca.fit(X_scaled)
```

---

# 26. What does `transform()` do?

**Answer:**

> `transform()` converts data into the already learned principal-component space.

Example:

```python
X_pca = pca.transform(X_scaled)
```

---

# 27. How do you handle new data with PCA?

**Answer:**

We use the already-fitted scaler and PCA.

```python
new_scaled = scaler.transform(new_data)

new_pca = pca.transform(new_scaled)
```

We do **not** fit them again on the new data.

---

# 28. Can PCA be used for visualization?

**Answer:**

Yes.

For example:

```text
50 features
    ↓ PCA
2 components
    ↓
2D visualization
```

We can plot:

```text
X-axis → PC1
Y-axis → PC2
```

This lets us visualize high-dimensional data in two dimensions.

---

# 29. Can PCA be used before K-Means?

**Answer:**

Yes.

Example:

```text
100 features
      ↓
     PCA
      ↓
20 components
      ↓
   K-Means
      ↓
   Clusters
```

But we should compare results with and without PCA rather than assuming PCA will improve clustering.

---

# 30. What is the difference between PCA and K-Means?

**Answer:**

```text
PCA
→ Reduces dimensions

K-Means
→ Finds clusters
```

Example:

```text
100 features
     ↓ PCA
20 components
     ↓ K-Means
5 clusters
```

PCA and K-Means solve different problems.

---

# 31. What are PCA's advantages?

**Answer:**

* Reduces dimensionality
* Can reduce redundant information
* Helps visualization
* Can reduce computational cost
* Can help when features are highly correlated

---

# 32. What are PCA's disadvantages?

**Answer:**

* Principal components can be difficult to interpret.
* Some information can be lost when components are removed.
* PCA is affected by feature scaling.
* PCA does not necessarily improve model performance.
* Components are combinations of original features, so business meaning can become less obvious.

---

# 33. Does PCA always improve a model?

**Answer:**

> No. PCA can help in some datasets, but it can also remove information that is useful for the particular task. Therefore, we should compare model performance with and without PCA.

---

# 34. Is PCA useful if we only have 3 features?

**Answer:**

> Not necessarily. If we already have only a few meaningful features, dimensionality reduction may not provide much benefit. We can still use PCA for learning, visualization, or if the features have useful correlation structure.

---

# 35. Can PCA be used with categorical data directly?

**Answer:**

> Standard PCA is designed for numerical features. Categorical features need appropriate encoding or another dimensionality-reduction method depending on the problem.

---

# 36. What are PCA loadings?

**Answer:**

> Loadings indicate how strongly the original features contribute to each principal component.

Example:

```text
PC1 =
0.7 × Income
+ 0.5 × Age
- 0.3 × Spending
```

The coefficients indicate the contribution of each original feature to PC1.

---

# 37. Interview scenario: You have 100 features. What will you do?

**Answer:**

> I would first inspect the data and preprocess it. If dimensionality is high and many features are correlated or redundant, I would consider PCA. I would standardize the numerical features, fit PCA, examine explained variance, and select an appropriate number of components. Then I would evaluate whether the reduced representation improves efficiency or downstream model performance.

---

# 38. Interview scenario: PCA gives 95% explained variance. Does that mean 95% accuracy?

**Answer:**

**No.**

> 95% explained variance means the selected components capture approximately 95% of the variance in the original feature space. It does not mean the model has 95% prediction accuracy.

---

# 39. Interview scenario: Why not just remove 95% of the columns?

**Answer:**

> Because PCA doesn't assume that entire original columns are unnecessary. Important information may be distributed across many features. PCA combines information from multiple original features into components and then retains the components that explain most of the variance.

---

# 40. ⭐ Perfect answer: "Explain PCA with an example"

> **"Suppose I have a customer dataset with Age, Annual Income and Spending Score. These are three dimensions. PCA analyzes the variation and relationships among these features and creates new features called principal components. PC1 captures the maximum variance, PC2 captures the next highest variance, and PC3 captures the remaining variance. Suppose PC1 explains 70%, PC2 explains 20%, and PC3 explains 10%. If I want to retain 90% of the variance, I can keep PC1 and PC2 and reduce the data from three dimensions to two. So PCA does not simply delete columns; it transforms the original features into new components and keeps fewer components while preserving most of the variance."**

---

# ⭐ One-line memory

```text
PCA

Original Features
      ↓
Find maximum-variance directions
      ↓
Create PC1, PC2, PC3...
      ↓
Check explained variance
      ↓
Keep required components
      ↓
Reduced dimensions
```

**Most important sentence for interview:**

> **"PCA transforms the original features into principal components ordered by the amount of variance they explain, allowing us to reduce dimensionality while retaining most of the important variation in the data."**



          ORIGINAL DATA
               ↓
        100 features
               ↓
       "Too many dimensions"
               ↓
              PCA
               ↓
      Create new features
               ↓
     PC1, PC2, PC3, ...
               ↓
     Check explained variance
               ↓
      Choose required PCs
               ↓
      Reduced feature set
               ↓
 ┌─────────────┼─────────────┐
 ↓             ↓             ↓
Model       Clustering   Visualization
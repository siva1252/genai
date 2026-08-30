Dataset
   ↓
No target/label
   ↓
Clustering problem
   ↓
Try K = 2, 3, 4, 5...
   ↓
For each K:
   ↓
K-Means learns K centroids
   ↓
Assign data points to nearest centroid
   ↓
Recalculate centroids
   ↓
Assign again
   ↓
Recalculate again
   ↓
Repeat until convergence
   ↓
Get final clusters
   ↓
Calculate Elbow + Silhouette
   ↓
Compare different K values
   ↓
Choose best K
   ↓
Suppose K = 3
   ↓
Final model has 3 centroids
   ↓
Evaluate cluster quality
   ↓
Interpret the 3 clusters
   ↓
Save trained model
   ↓
NEW DATA
   ↓
Same preprocessing/scaling
   ↓
model.predict()
   ↓
Calculate distance to existing 3 centroids
   ↓
Nearest centroid
   ↓
Return cluster ID


You can remember K-Means as:

K-Means' main job is to find K clusters and their centroids. Elbow/Silhouette is used to decide what K should be.



# K-MEANS CLUSTERING — COMPLETE INTERVIEW GUIDE

## 1. What is K-Means?

**K-Means is an unsupervised machine learning clustering algorithm.**

It groups similar data points into **K clusters** based on the distance between data points and cluster centroids.

The important point is:

```text
K-Means → Unsupervised Learning
       → No target/label (y)
       → Finds groups automatically
```

### Simple example

Suppose we have customer data:

| Customer | Age | Income | Spending Score |
| -------- | --: | -----: | -------------: |
| A        |  22 |    25K |             30 |
| B        |  25 |    30K |             35 |
| C        |  50 |    90K |             80 |
| D        |  52 |    95K |             85 |
| E        |  40 |    60K |             55 |

We don't have a column saying:

```text
Cluster = ?
```

K-Means discovers the groups.

For example:

```text
Cluster 0 → Young / low income customers
Cluster 1 → High income / high spending customers
Cluster 2 → Medium income / medium spending customers
```

---

# 2. Why do we use K-Means?

We use K-Means when we want to **discover natural groups in unlabeled data**.

### Examples

* Customer segmentation
* Grouping products
* Grouping users
* Market segmentation
* Document clustering
* Image segmentation
* Finding similar observations

### Interview answer

> "K-Means is used when we have unlabeled data and want to divide the data into groups based on similarity."

---

# 3. How do I know whether to use clustering?

This is one of the most important things.

First ask:

### Do I have a target variable?

```text
                Dataset
                   ↓
             Do I have y?
              /         \
            YES          NO
             ↓            ↓
       Supervised     Unsupervised
```

If there is no target:

```text
Unsupervised
     ↓
What is the requirement?
     ↓
Need groups?
     → Clustering

Need fewer features?
     → Dimensionality Reduction
```

### Example

Question:

> "Group customers based on their behavior."

Answer:

```text
Clustering
```

Question:

> "Reduce 20 features into 2 or 3 features."

Answer:

```text
Dimensionality Reduction
```

---

# 4. Why is K-Means called K-Means?

There are two important words.

### K

`K` means:

> Number of clusters we want.

For example:

```text
K = 3
```

means:

```text
3 clusters
```

### Means

K-Means calculates the **mean/average position of points in each cluster**.

That average position becomes the **centroid**.

So:

```text
K-Means
   ↓
K = number of clusters
Means = average position → centroid
```

---

# 5. What is a centroid?

A centroid is the **center/mean position of a cluster**.

Suppose a cluster contains:

```text
Age: 20, 22, 24
```

The mean is:

```text
(20 + 22 + 24) / 3 = 22
```

So the centroid's Age coordinate is `22`.

With multiple features, the centroid has one coordinate for every feature.

---

# 6. Very important: K = number of centroids

If:

```text
K = 3
```

K-Means learns:

```text
3 clusters
+
3 centroids
```

So:

```text
K = 3
→ 3 clusters
→ 3 centroids
```

You **cannot have K = 10 and only 4 centroids** in standard K-Means.

```text
K = 10
→ 10 clusters
→ 10 centroids
```

---

# 7. Do we directly know K?

No.

Usually we don't simply say:

```text
K = 3
```

without checking.

We try different K values.

For example:

```text
K = 2
K = 3
K = 4
K = 5
...
K = 10
```

Then evaluate them.

The main methods are:

```text
Elbow Method
+
Silhouette Score
```

---

# 8. How do we choose the range of K?

K is **not directly equal to the number of rows**.

If we have:

```text
10,000 rows
```

we don't normally test:

```text
K = 1 → 10,000
```

Instead, the developer chooses a **reasonable range** based on the dataset, business problem, expected groups, computational cost, and exploratory analysis.

For example:

```text
K = 2 → 10
```

or:

```text
K = 2 → 15
```

There is no universal rule saying:

> "10,000 rows means test 10,000 K values."

---

# 9. Elbow Method

The Elbow Method uses **inertia**.

### What is inertia?

Inertia measures the total squared distance between each data point and its assigned centroid.

Conceptually:

```text
Point
  ↓
Distance to its centroid
  ↓
Squared distance
  ↓
Sum for all points
  ↓
Inertia
```

Lower inertia is better.

But there is a problem:

> Increasing K almost always decreases inertia.

Therefore, we don't simply choose the smallest inertia.

We look for the **elbow point** where increasing K further gives much smaller improvement.

Example:

```text
K       Inertia

2       900
3       500
4       350
5       320
6       310
7       305
```

The major improvement happens around:

```text
K = 3 or 4
```

We investigate that region further.

---

# 10. Silhouette Score

Silhouette Score evaluates how well each point fits its own cluster compared with other clusters.

Range:

```text
-1 → +1
```

Generally:

```text
Higher → better
```

Interpretation:

```text
Close to +1
→ well-separated clusters

Around 0
→ overlapping clusters

Negative
→ possible poor assignment
```

We calculate Silhouette for different K values.

Example:

```text
K       Silhouette

2       0.51
3       0.64
4       0.58
5       0.49
```

Here:

```text
K = 3
```

has the highest Silhouette Score.

---

# 11. Elbow + Silhouette together

We shouldn't blindly depend on only one method.

We can do:

```text
Try K values
      ↓
Elbow Method
      ↓
Silhouette Score
      ↓
Compare
      ↓
Choose suitable K
```

For example:

```text
K = 2 → Silhouette 0.51
K = 3 → Silhouette 0.64
K = 4 → Silhouette 0.58
K = 5 → Silhouette 0.49
```

and the elbow also appears around `K = 3`.

Then:

```text
Final K = 3
```

---

# 12. Important clarification about centroids

We don't first calculate the final centroids and then decide K.

Correct flow:

```text
Dataset
   ↓
Try K = 2
   ↓
K-Means fit()
   ↓
2 centroids
   ↓
Calculate clusters
   ↓
Evaluate

Try K = 3
   ↓
K-Means fit()
   ↓
3 centroids
   ↓
Calculate clusters
   ↓
Evaluate

Try K = 4
   ↓
K-Means fit()
   ↓
4 centroids
   ↓
Calculate clusters
   ↓
Evaluate
```

Then:

```text
Elbow + Silhouette
       ↓
Choose final K
```

---

# 13. Does `fit()` calculate everything?

**Yes.**

We do NOT manually calculate:

```text
Distances
Centroids
Assignments
Recalculation
Convergence
```

`KMeans.fit()` does that internally.

For example:

```python
model = KMeans(n_clusters=3)

model.fit(X_scaled)
```

Internally, K-Means performs the algorithm.

---

# 14. How K-Means works internally

Suppose:

```text
K = 3
```

### Step 1 — Initialize centroids

K-Means starts with 3 centroid positions.

```text
Centroid 1
Centroid 2
Centroid 3
```

### Step 2 — Calculate distance

Each point's distance to each centroid is calculated.

For example:

```text
Point A

Distance → Centroid 1 = 2.1
Distance → Centroid 2 = 7.5
Distance → Centroid 3 = 4.2
```

Nearest:

```text
Centroid 1
```

So Point A belongs to Cluster 1.

### Step 3 — Assign all points

Every point is assigned to its nearest centroid.

### Step 4 — Recalculate centroids

For each cluster, K-Means calculates the mean position.

### Step 5 — Assign again

Points are assigned again based on the new centroids.

### Step 6 — Repeat

```text
Assign
 ↓
Recalculate centroid
 ↓
Assign again
 ↓
Recalculate
 ↓
Repeat
```

until the algorithm converges.

---

# 15. Final K-Means flow

```text
Dataset
   ↓
Select features X
   ↓
Scale features
   ↓
Try different K values
   ↓
K-Means fit() for each K
   ↓
Elbow + Silhouette
   ↓
Choose best/suitable K
   ↓
Create final K-Means model
   ↓
model.fit(X_scaled)
   ↓
Final centroids
+
Final cluster labels
   ↓
Evaluate clusters
   ↓
Interpret clusters
   ↓
Save trained model
```

---

# 16. Why do we scale K-Means data?

K-Means uses **distance**.

Suppose:

```text
Age = 25
Income = 90000
```

Income has much larger numerical values.

Without scaling, Income can dominate the distance calculation.

Therefore:

```python
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

Now features are on a comparable scale.

### Interview answer

> "Because K-Means is distance-based, features with larger numerical scales can dominate the distance calculation. Scaling prevents that."

---

# 17. Does K-Means always require scaling?

Not necessarily.

If all features are already on comparable scales, scaling may not be necessary.

But when features have very different ranges, scaling is generally important.

---

# 18. Why don't we use `y`?

Because K-Means is unsupervised.

We have:

```python
X
```

but not:

```python
y
```

So:

```python
model.fit(X_scaled)
```

instead of:

```python
model.fit(X_train, y_train)
```

---

# 19. Do we use train/test split?

For basic K-Means clustering, we generally don't use the same `X_train`, `X_test`, `y_train`, `y_test` workflow as supervised learning.

Why?

Because there is no known target `y` to compare predictions against.

Typical workflow:

```text
X
 ↓
Scale
 ↓
K-Means
 ↓
Clusters
```

For new data:

```text
New X
 ↓
Same scaler
 ↓
model.predict()
 ↓
Cluster
```

---

# 20. What happens after choosing K?

Suppose:

```text
Best K = 3
```

Now create the final model:

```python
model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)
```

Then:

```python
model.fit(X_scaled)
```

Now the model learns the final:

```text
3 centroids
+
cluster assignments
```

---

# 21. What is `model.labels_`?

After fitting:

```python
model.fit(X_scaled)
```

we can get:

```python
model.labels_
```

It contains the cluster assigned to each training data point.

Example:

```text
[0, 0, 2, 1, 1, 2, 0]
```

Meaning:

```text
Point 1 → Cluster 0
Point 2 → Cluster 0
Point 3 → Cluster 2
Point 4 → Cluster 1
...
```

---

# 22. What is `model.cluster_centers_`?

It gives the learned centroids.

```python
model.cluster_centers_
```

If:

```text
K = 3
```

there will be:

```text
3 centroids
```

and each centroid has one value for each feature.

---

# 23. `fit()` vs `predict()`

### `fit()`

```python
model.fit(X_scaled)
```

Means:

> Learn the clusters and centroids from the existing data.

### `predict()`

```python
model.predict(new_X_scaled)
```

Means:

> Assign new data to the nearest already-learned centroid.

---

# 24. What happens when new data arrives?

Suppose the trained model has:

```text
Centroid 0
Centroid 1
Centroid 2
```

A new customer arrives.

```text
Age = 50
Income = 90000
Spending = 80
```

We first use the same scaler:

```python
new_X_scaled = scaler.transform(new_X)
```

Then:

```python
prediction = model.predict(new_X_scaled)
```

The model calculates the distance from the new point to the existing centroids.

```text
New customer
     ↓
Distance to C0
Distance to C1
Distance to C2
     ↓
Nearest centroid
     ↓
Cluster
```

---

# 25. Do we run Elbow again for new data?

**No.**

Once the production model has been trained with the selected K:

```text
K = 3
```

we don't run:

```text
Elbow
Silhouette
K selection
```

for every new customer.

We simply:

```text
New data
 ↓
Same scaler.transform()
 ↓
model.predict()
 ↓
Cluster
```

---

# 26. What if new data comes through CSV?

Suppose:

```text
new_customers.csv
```

contains:

```text
Age,Annual_Income,Spending_Score
28,35000,35
45,70000,60
52,95000,85
```

Read it:

```python
new_df = pd.read_csv("new_customers.csv")
```

Select features:

```python
new_X = new_df[
    ["Age", "Annual_Income", "Spending_Score"]
]
```

Scale:

```python
new_X_scaled = scaler.transform(new_X)
```

Predict:

```python
predictions = model.predict(new_X_scaled)
```

Add results:

```python
new_df["Cluster"] = predictions

print(new_df)
```

---

# 27. Why `transform()` and not `fit_transform()` for new data?

Training:

```python
scaler.fit_transform(X)
```

The scaler learns the scaling parameters from training data.

New data:

```python
scaler.transform(new_X)
```

We use the **same learned scaling parameters**.

Don't do:

```python
scaler.fit_transform(new_X)
```

because that creates new scaling parameters based on the new data.

---

# 28. Main K-Means metric — Silhouette Score

```python
silhouette_score(
    X_scaled,
    model.labels_
)
```

Higher is generally better.

It measures:

```text
How well a point fits its own cluster
+
How separated it is from other clusters
```

---

# 29. Inertia

K-Means provides:

```python
model.inertia_
```

Lower inertia means points are closer to their assigned centroids.

But:

> We don't choose the model simply because it has the lowest inertia.

Because increasing K generally reduces inertia.

That's why we use the **Elbow Method**.

---

# 30. Other clustering metrics

When comparing clustering algorithms, useful internal metrics include:

### Silhouette Score

```text
Higher → better
```

### Davies-Bouldin Index

```text
Lower → better
```

### Calinski-Harabasz Index

```text
Higher → better
```

---

# 31. Comparing K-Means with other clustering algorithms

Suppose we use:

```text
K-Means
Hierarchical Clustering
DBSCAN
```

We can compare:

| Model        | Silhouette | Davies-Bouldin | Calinski-Harabasz |
| ------------ | ---------: | -------------: | ----------------: |
| K-Means      |       0.63 |           0.52 |               450 |
| Hierarchical |       0.59 |           0.61 |               410 |
| DBSCAN       |       0.48 |           0.75 |               300 |

Remember:

```text
Silhouette          → Higher ↑
Davies-Bouldin      → Lower ↓
Calinski-Harabasz   → Higher ↑
```

But there is **no universal single metric that always determines the best clustering algorithm**.

We also consider:

```text
Metrics
+
Cluster shape
+
Outliers/noise
+
Business meaning
+
Interpretability
```

---

# 32. K-Means complete code

```python
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)


# 1. Load dataset

df = pd.read_csv("customer_clustering.csv")


# 2. Select features

X = df[
    [
        "Age",
        "Annual_Income",
        "Spending_Score"
    ]
]


# 3. Scale features

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# 4. Find suitable K

results = []

for k in range(2, 11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    labels = model.labels_

    results.append({
        "K": k,
        "Inertia": model.inertia_,
        "Silhouette": silhouette_score(
            X_scaled,
            labels
        )
    })


results_df = pd.DataFrame(results)

print(results_df)


# 5. Select K after checking
#    Elbow + Silhouette

best_k = 3


# 6. Create final model

model = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)


# 7. Train final model

model.fit(X_scaled)


# 8. Get cluster labels

df["Cluster"] = model.labels_


# 9. Evaluate final model

silhouette = silhouette_score(
    X_scaled,
    model.labels_
)

davies_bouldin = davies_bouldin_score(
    X_scaled,
    model.labels_
)

calinski_harabasz = calinski_harabasz_score(
    X_scaled,
    model.labels_
)


print("Silhouette:", silhouette)
print("Davies-Bouldin:", davies_bouldin)
print("Calinski-Harabasz:", calinski_harabasz)


# 10. Final centroids

print("Centroids:")
print(model.cluster_centers_)


# 11. Cluster assignments

print(df.head())
```

---

# 33. The code flow you should remember

This is the most important part for interviews:

```text
Load Dataset
     ↓
Select X
     ↓
No y
     ↓
Scale X
     ↓
Try different K values
     ↓
KMeans.fit() for each K
     ↓
Inertia + Silhouette
     ↓
Elbow + Silhouette
     ↓
Choose suitable K
     ↓
Create final KMeans model
     ↓
model.fit(X_scaled)
     ↓
model.labels_
+
model.cluster_centers_
     ↓
Evaluate clusters
     ↓
Interpret clusters
     ↓
Save model
     ↓
New data arrives
     ↓
scaler.transform()
     ↓
model.predict()
     ↓
Cluster
```

---

# K-MEANS INTERVIEW QUESTIONS & ANSWERS

## Q1. What is K-Means?

**Answer:**

> K-Means is an unsupervised clustering algorithm that divides unlabeled data into K clusters by assigning each point to the nearest centroid and iteratively updating the centroids until convergence.

---

## Q2. Why is K-Means unsupervised?

**Answer:**

> Because K-Means does not require a target variable or predefined labels. It discovers groups from the feature data itself.

---

## Q3. What does K represent?

**Answer:**

> K represents the number of clusters we want the algorithm to create.

---

## Q4. If K = 5, how many centroids are created?

**Answer:**

> Five clusters and five centroids.

---

## Q5. How do you choose K?

**Answer:**

> I test a reasonable range of K values and use methods such as the Elbow Method and Silhouette Score to determine a suitable K.

---

## Q6. What is the Elbow Method?

**Answer:**

> The Elbow Method plots inertia against different K values. We look for the point where increasing K further gives significantly smaller improvement in inertia.

---

## Q7. What is inertia?

**Answer:**

> Inertia is the sum of squared distances between each data point and its assigned cluster centroid.

---

## Q8. Is lower inertia always better?

**Answer:**

> Lower inertia is better in isolation, but we cannot simply choose the K with the lowest inertia because inertia generally decreases as K increases. That's why we use the Elbow Method.

---

## Q9. What is Silhouette Score?

**Answer:**

> Silhouette Score measures how well each data point fits within its assigned cluster compared with the nearest other cluster.

---

## Q10. What is a good Silhouette Score?

**Answer:**

> Scores closer to 1 generally indicate better-separated and more compact clusters. Around zero indicates overlapping clusters, while negative values can indicate poor assignments.

---

## Q11. Why do we scale data before K-Means?

**Answer:**

> K-Means is distance-based. If features have very different scales, larger-scale features can dominate the distance calculation. Scaling puts features on a comparable scale.

---

## Q12. Does K-Means always require StandardScaler?

**Answer:**

> No. It depends on the feature scales. If features are already comparable, scaling may not be necessary. But when feature ranges differ significantly, scaling is usually important.

---

## Q13. What does `fit()` do in K-Means?

**Answer:**

> `fit()` performs the K-Means learning process: it initializes centroids, assigns points, recalculates centroids, and repeats until convergence.

---

## Q14. Are we manually calculating centroids in Python?

**Answer:**

> No. Scikit-learn's `KMeans.fit()` performs the centroid initialization, assignment, recalculation, and convergence internally.

---

## Q15. Why do we write a loop for different K values?

**Answer:**

> Because K is a hyperparameter and K-Means doesn't automatically know the appropriate number of clusters. We train models with different K values and evaluate them.

---

## Q16. Do we run `fit()` for every K?

**Answer:**

> Yes. Each K represents a different clustering configuration, so we fit a separate K-Means model for each candidate K.

---

## Q17. After selecting K = 3, why do we call `fit()` again?

**Answer:**

> The earlier fits were used to evaluate candidate K values. After choosing K = 3, we create the final K-Means model with K = 3 and fit it to obtain the final centroids and cluster assignments.

---

## Q18. What is `model.labels_`?

**Answer:**

> It contains the cluster assignment for each data point used during fitting.

---

## Q19. What is `model.cluster_centers_`?

**Answer:**

> It contains the learned centroid coordinates for each cluster.

---

## Q20. How do you predict a new data point?

**Answer:**

> I first apply the same fitted scaler using `transform()`, then pass the scaled data to `model.predict()`. The model assigns it to the nearest learned centroid.

---

## Q21. Do you calculate K again when new data arrives?

**Answer:**

> No. Once the final K is selected and the model is trained, new data is assigned to the existing clusters using the existing centroids.

---

## Q22. Do you run Elbow and Silhouette for every new customer?

**Answer:**

> No. Those methods are used during model development to select and evaluate the clustering configuration, not for every new prediction.

---

## Q23. Why use `transform()` instead of `fit_transform()` on new data?

**Answer:**

> Because the new data must use the same scaling parameters learned from the training data. Using `fit_transform()` would calculate new scaling parameters from the new data.

---

## Q24. Does K-Means use train/test split like regression?

**Answer:**

> Not in the same standard way. Since there is no target variable, there is no `y_test` for direct supervised evaluation. Basic K-Means usually fits on the available feature data and uses internal clustering metrics for evaluation.

---

## Q25. What metrics are used for clustering?

**Answer:**

> Common internal clustering metrics include Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index.

---

## Q26. Which direction is better for these metrics?

**Answer:**

```text
Silhouette Score          → Higher is better
Davies-Bouldin Index      → Lower is better
Calinski-Harabasz Index   → Higher is better
```

---

## Q27. Can we compare K-Means with Hierarchical and DBSCAN?

**Answer:**

> Yes. We can run the algorithms on suitable versions of the same feature data and compare clustering-quality metrics, while also considering cluster shape, noise, outliers, and business interpretation.

---

## Q28. What is the main limitation of K-Means?

**Answer:**

> K-Means requires us to choose K beforehand and works best when clusters are relatively compact and well separated. It can also be sensitive to initialization, feature scaling, and outliers.

---

## Q29. How does K-Means handle outliers?

**Answer:**

> K-Means can be sensitive to outliers because centroids are based on means. Extreme points can pull the centroid away from the main group.

---

## Q30. What is `n_init`?

**Answer:**

> `n_init` controls how many different centroid initializations K-Means tries. The algorithm keeps the result with the best objective value, typically the lowest inertia among those runs.

---

## Q31. Why use `random_state`?

**Answer:**

> It makes the initialization reproducible, so we can obtain consistent results when running the same code again.

---

## Q32. What distance does K-Means commonly use?

**Answer:**

> Standard K-Means commonly uses Euclidean distance when assigning points to centroids.

---

## Q33. Can K-Means work with categorical data directly?

**Answer:**

> Standard K-Means is designed for numerical feature spaces and distance calculations. Pure categorical data should not simply be treated as ordinary numeric values; an appropriate encoding or another clustering algorithm designed for categorical/mixed data may be more suitable.

---

## Q34. What happens if K is too small?

**Answer:**

> Different natural groups may be merged together, producing overly broad clusters.

---

## Q35. What happens if K is too large?

**Answer:**

> A natural group may be unnecessarily split into several smaller clusters, making the clustering less meaningful.

---

## Q36. Explain your K-Means project from start to finish.

**Strong interview answer:**

> "First, I identified that the problem was an unsupervised clustering problem because there was no target variable and the requirement was to discover groups. I selected the relevant numerical features and scaled them because K-Means is distance-based. Since K is a hyperparameter, I tested a reasonable range of K values. For each K, I fitted a K-Means model and evaluated inertia and Silhouette Score. I used the Elbow Method together with Silhouette Score to select a suitable K. After selecting the final K, I created the final K-Means model and called `fit()` on the scaled data. The model learned the final centroids and cluster assignments. I evaluated the final clustering using Silhouette Score and other clustering metrics and interpreted the characteristics of each cluster. When new data arrives, I apply the same scaler using `transform()` and use `model.predict()` to assign the new observation to the nearest existing centroid."

---

# ONE-LINE MEMORY

```text
K-Means = X only
        ↓
Scale
        ↓
Find K
        ↓
Elbow + Silhouette
        ↓
Final K
        ↓
fit()
        ↓
Centroids + Labels
        ↓
Evaluate
        ↓
New data → transform() → predict()
```

This is the **complete K-Means flow you should be able to explain in an interview**.

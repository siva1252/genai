# Unsupervised Learning — Complete Interview Guide

If you learn this document properly, you should be able to explain **Unsupervised Learning, K-Means, Hierarchical Clustering, DBSCAN, preprocessing, cluster selection, metrics, model comparison, and project implementation** in an interview.

---

# 1. What is Machine Learning?

Machine Learning is a technique where a computer learns patterns from data and uses those patterns to make decisions or predictions.

There are three major types:

```text
Machine Learning
│
├── Supervised Learning
│   ├── Regression
│   └── Classification
│
├── Unsupervised Learning
│   ├── Clustering
│   └── Dimensionality Reduction
│
└── Reinforcement Learning
```

---

# 2. What is Supervised Learning?

In supervised learning, the dataset contains:

```text
X → Input features
y → Target/output
```

Example:

```text
Age   Income   →   Customer will buy?
25    40K          Yes
35    80K          No
```

The model learns:

```text
X → y
```

Examples:

* Linear Regression
* Logistic Regression
* Decision Tree
* Random Forest
* SVM

---

# 3. What is Unsupervised Learning?

In unsupervised learning, we **don't have a target variable `y`**.

We only have:

```text
X → Features
```

Example:

```text
Age
Annual Income
Spending Score
```

We don't tell the model:

```text
Customer 1 → Cluster 0
Customer 2 → Cluster 1
```

Instead, we ask the algorithm to discover patterns itself.

```text
X
↓
Find patterns
↓
Find similarities/differences
↓
Groups / structure
```

### Interview answer

> **Unsupervised learning is a type of machine learning where the data does not have labelled target values, and the algorithm learns the underlying structure or patterns from the input data itself.**

---

# 4. Why do we use Unsupervised Learning?

Main purposes:

### 1. Clustering

Find similar groups.

Example:

```text
Customers
   ↓
Customer segments
```

### 2. Dimensionality Reduction

Reduce many features into fewer dimensions while preserving useful information.

Example:

```text
100 features
     ↓
PCA
     ↓
10 components
```

### 3. Anomaly / Noise Detection

Find unusual observations.

Example:

```text
Normal transactions
        ↓
Dense pattern

Unusual transaction
        ↓
Possible anomaly
```

---

# 5. What is Clustering?

Clustering is the process of grouping **similar data points together** and separating dissimilar points.

Example:

```text
Customer Data
      ↓
Similarity
      ↓
Cluster 0
Cluster 1
Cluster 2
```

The important point is:

> We don't already know the correct cluster labels.

---

# 6. What does "similar" mean?

Similarity usually comes from a **distance or similarity measure**.

For example, Euclidean distance:

```text
distance = √((x₁-x₂)² + (y₁-y₂)²)
```

If two customers have similar:

```text
Age
Income
Spending Score
```

their distance can be small.

If their characteristics are very different, their distance can be large.

---

# 7. Main Clustering Algorithms

The three algorithms we studied are:

```text
Clustering
│
├── K-Means
│
├── Hierarchical Clustering
│
└── DBSCAN
```

Their fundamental ideas are different.

```text
K-Means
→ Centroid-based

Hierarchical
→ Distance + merging

DBSCAN
→ Density-based
```

---

# 8. K-Means

K-Means divides data into **K clusters**.

Example:

```text
K = 3

Customer Data
      ↓
Cluster 0
Cluster 1
Cluster 2
```

---

# 9. Why is it called K-Means?

Because:

* `K` = number of clusters
* `Means` = mean/average point, which becomes the centroid

---

# 10. K-Means workflow

```text
Dataset
   ↓
Select features
   ↓
Scale data
   ↓
Choose K
   ↓
Create K centroids
   ↓
Assign points to nearest centroid
   ↓
Calculate new centroids
   ↓
Reassign points
   ↓
Repeat
   ↓
Final clusters
```

---

# 11. What happens inside `KMeans.fit()`?

Suppose:

```python
model = KMeans(n_clusters=3)
```

Then:

```python
model.fit(X_scaled)
```

internally performs the K-Means clustering process.

Conceptually:

```text
Choose initial centroids
       ↓
Calculate distance
       ↓
Assign each point to nearest centroid
       ↓
Calculate new mean/centroid
       ↓
Reassign points
       ↓
Repeat until convergence
```

You don't manually calculate every distance.

---

# 12. Why do we scale data for K-Means?

K-Means is distance-based.

Suppose:

```text
Age             20–60
Income          20,000–150,000
Spending Score  1–100
```

Income has much larger numerical values.

Without scaling, it can dominate the distance calculation.

So:

```python
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

---

# 13. How do we choose K?

One common technique is the **Elbow Method**.

We test:

```text
K = 2
K = 3
K = 4
K = 5
...
```

and calculate the within-cluster sum of squares/inertia.

Example:

```text
K       Inertia

2       500
3       300
4       220
5       200
6       190
```

We plot the values and look for an **elbow** where the improvement begins to reduce significantly.

---

# 14. What is inertia?

Inertia measures how close points are to their assigned cluster centroid.

Lower inertia means points are closer to their centroids.

But:

> Inertia will generally decrease as K increases.

Therefore, we should **not simply choose the K with the lowest inertia**.

We use the elbow along with other evidence.

---

# 15. Can we use Silhouette Score to choose K?

Yes.

We can test different K values and calculate Silhouette Score.

```text
Higher Silhouette
→ generally better separated and cohesive clusters
```

So we can use:

```text
Elbow
+
Silhouette
+
Domain/business meaning
```

to choose K.

---

# 16. What is a centroid?

A centroid is the mean position of all points assigned to a cluster.

Example:

```text
Cluster:

(2,4)
(4,6)
(6,8)

Centroid:

(4,6)
```

The centroid represents the center of that cluster.

---

# 17. How does K-Means predict new data?

This is one of the biggest advantages of K-Means.

After training:

```python
model.fit(X_scaled)
```

the model has learned cluster centroids.

For new data:

```python
new_data_scaled = scaler.transform(new_data)

prediction = model.predict(new_data_scaled)
```

It calculates the distance between the new point and each centroid.

```text
New customer
     ↓
Distance to Cluster 0 centroid
Distance to Cluster 1 centroid
Distance to Cluster 2 centroid
     ↓
Nearest centroid
     ↓
Assigned cluster
```

---

# 18. Why do we use `scaler.transform()` for new data instead of `fit_transform()`?

This is very important.

During training:

```python
X_scaled = scaler.fit_transform(X)
```

The scaler **learns parameters from training data** and transforms it.

For new data:

```python
new_data_scaled = scaler.transform(new_data)
```

We use the **same already-fitted scaler**.

We should not do:

```python
scaler.fit_transform(new_data)
```

because that would calculate new scaling parameters from the new data.

---

# 19. K-Means advantages

* Simple
* Fast
* Easy to understand
* Works well for reasonably separated spherical/compact clusters
* Can assign new data using `predict()`

---

# 20. K-Means disadvantages

* Need to choose K
* Sensitive to outliers
* Sensitive to feature scaling
* Can struggle with irregular cluster shapes
* Results can depend on initialization

---

# 21. Hierarchical Clustering

Hierarchical clustering creates a **hierarchy of clusters**.

The common approach we studied is **Agglomerative Clustering**.

It starts with:

```text
Every customer = separate cluster
```

and repeatedly merges the closest clusters.

---

# 22. Hierarchical workflow

```text
Every point
   ↓
Calculate distances
   ↓
Find closest points/clusters
   ↓
Merge
   ↓
Calculate new cluster distances
   ↓
Merge again
   ↓
Continue
   ↓
Dendrogram
   ↓
Choose final clusters
```

---

# 23. What is a dendrogram?

A dendrogram is a tree-like diagram showing **how clusters are progressively merged and at what distances**.

Example:

```text
        ┌──── A
    ┌───┤
    │   └──── B
────┤
    │   ┌──── C
    └───┤
        └──── D
```

The height where two groups merge represents the distance according to the selected linkage method.

---

# 24. How do we choose the number of clusters in Hierarchical Clustering?

We inspect the dendrogram and make a **horizontal cut**.

Example:

```text
        │
    ┌───┴───┐
    │       │
 ┌──┴──┐ ┌──┴──┐
 A     B C     D
```

Draw a horizontal line.

The number of branches crossed gives the number of clusters.

---

# 25. What is linkage?

Linkage defines **how the distance between two clusters is calculated**.

Common methods:

```text
Single linkage
Complete linkage
Average linkage
Ward linkage
```

---

# 26. What is Ward linkage?

Ward linkage merges clusters based on the increase in **within-cluster variance** / sum of squared deviations.

It generally tends to create compact clusters.

Example:

```python
z = linkage(
    X_scaled,
    method="ward"
)
```

Then:

```python
model = AgglomerativeClustering(
    n_clusters=3,
    metric="euclidean",
    linkage="ward"
)
```

---

# 27. What is the difference between Single, Complete and Average linkage?

### Single

Uses the minimum distance between points in two clusters.

```text
Cluster A ● ●

Cluster B        ● ●

Shortest pair → determines distance
```

### Complete

Uses the maximum distance.

### Average

Uses the average pairwise distance.

### Ward

Looks at the increase in within-cluster variance.

---

# 28. Does Hierarchical clustering require the number of clusters?

It depends on how you use it.

The **hierarchy itself** does not require you to know the final number of clusters beforehand.

You can create the hierarchy/dendrogram and decide where to cut it.

In scikit-learn's `AgglomerativeClustering`, you can specify:

```python
n_clusters=3
```

when you want the final 3-cluster solution.

---

# 29. Why use Hierarchical Clustering?

Its major advantage is that it helps us understand:

> **How individual points and groups are related to one another at different distance levels.**

This is especially useful when you want to explore the structure of the dataset.

---

# 30. Does AgglomerativeClustering have normal `predict()`?

Standard sklearn `AgglomerativeClustering` does not provide a normal `predict()` method for arbitrary new observations.

This is different from K-Means.

```text
K-Means
→ fit()
→ predict(new data) ✅

Agglomerative
→ fit_predict()
→ normal predict(new data) ❌
```

---

# 31. Why is Hierarchical still useful without `predict()`?

Because its goal can be **exploratory structure discovery**.

For example:

```text
Customer data
      ↓
Distance relationships
      ↓
Hierarchy
      ↓
Dendrogram
      ↓
Natural group structure
```

The relationship structure itself can be valuable.

---

# 32. DBSCAN

DBSCAN stands for:

> **Density-Based Spatial Clustering of Applications with Noise**

It groups points based on **density**.

It can also identify noise/outliers.

---

# 33. Why use DBSCAN?

DBSCAN is useful when:

* Number of clusters is unknown.
* Clusters may have irregular shapes.
* Data contains noise/outliers.
* We want to find dense regions.

---

# 34. DBSCAN's two main parameters

```python
DBSCAN(
    eps=0.38,
    min_samples=6
)
```

### `eps`

Neighborhood radius.

### `min_samples`

Minimum number of samples required in that neighborhood for a point to qualify as a core point.

---

# 35. What is a Core Point?

A core point has the required number of samples in its `eps` neighborhood.

Conceptually:

```text
        ●
    ●   A   ●
      ● ●
        ●

A → Core
```

---

# 36. What is a Border Point?

A border point does not itself satisfy the core-point density requirement but lies within the neighborhood of a core point.

```text
● ● ●
 ● A ●       X

A → Core
X → Border
```

---

# 37. What is a Noise Point?

A point that is not a core point and is not density-reachable from a core point can be classified as noise.

In sklearn:

```text
-1 → Noise
```

---

# 38. How does DBSCAN create clusters?

Suppose:

```text
eps = 0.38
min_samples = 6
```

DBSCAN conceptually does:

```text
Point
 ↓
Find points inside eps
 ↓
Count samples
 ↓
Enough samples?
 ↓
Core point
 ↓
Expand through density-connected core points
 ↓
Add reachable border points
 ↓
Unconnected points → Noise
```

This happens internally during:

```python
labels = model.fit_predict(X_scaled)
```

---

# 39. What does "density-connected" mean?

Suppose:

```text
A → B → C
```

and the points are connected through density-reachable core points.

DBSCAN can place them in the same cluster even when A and C aren't directly within one another's neighborhood.

So:

> DBSCAN clusters are based on connected dense regions, not simply on every point being directly close to every other point.

---

# 40. How do we choose `eps`?

A common approach is the **K-distance plot**.

Process:

```text
Choose starting min_samples
        ↓
Find kth-nearest-neighbor distances
        ↓
Sort distances
        ↓
Plot distances
        ↓
Look for elbow/bend
        ↓
Candidate eps
```

The elbow provides a **candidate**, not a guaranteed perfect value.

---

# 41. How do we choose `min_samples`?

There is no universal correct value.

We choose a reasonable starting value based on the dataset and then tune it.

For a 3-feature dataset:

```text
Age
Income
Spending Score
```

we might start with:

```python
min_samples = 6
```

Then test values such as:

```text
4
5
6
7
8
```

---

# 42. What if `eps` is too small?

The neighborhood becomes too small.

Possible result:

```text
Many points → Noise
Small clusters
```

---

# 43. What if `eps` is too large?

The neighborhood becomes too large.

Possible result:

```text
Different groups
      ↓
Become connected
      ↓
Clusters merge
```

---

# 44. What if `min_samples` is too high?

The density requirement becomes stricter.

Possible result:

```text
Fewer core points
      ↓
More noise
```

---

# 45. What if `min_samples` is too low?

The density requirement becomes easier.

Possible result:

```text
More core points
      ↓
More connectivity
      ↓
Potentially merged clusters
```

---

# 46. Does DBSCAN require the number of clusters?

No.

You don't write:

```python
DBSCAN(n_clusters=3)
```

Instead:

```python
DBSCAN(
    eps=0.38,
    min_samples=6
)
```

DBSCAN discovers the number of clusters from the density structure.

---

# 47. Does DBSCAN have `predict()`?

Standard sklearn DBSCAN does **not provide a normal `predict()` method for arbitrary new data**.

We generally use:

```python
labels = model.fit_predict(X_scaled)
```

for the data being clustered.

---

# 48. Why is DBSCAN useful without `predict()`?

Because its purpose is often:

```text
Existing data
     ↓
Find dense groups
     +
Find noise/outliers
```

For example:

* anomaly detection
* spatial clustering
* irregular customer groups
* noisy datasets
* discovering unknown groups

---

# 49. K-Means vs Hierarchical vs DBSCAN

This is one of the **most important interview tables**.

| Feature                  | K-Means             | Hierarchical                        | DBSCAN               |
| ------------------------ | ------------------- | ----------------------------------- | -------------------- |
| Basic idea               | Centroid            | Distance + hierarchy                | Density              |
| Need number of clusters? | Yes                 | Can choose from hierarchy           | No                   |
| Main parameters          | `n_clusters`        | linkage, distance, `n_clusters`/cut | `eps`, `min_samples` |
| Dendrogram               | No                  | Yes                                 | No                   |
| Noise detection          | Poor                | Not its main purpose                | Yes                  |
| Irregular shapes         | Limited             | Depends on linkage                  | Good                 |
| New-data `predict()`     | Yes                 | Standard sklearn: No                | Standard sklearn: No |
| Sensitive to scaling     | Yes                 | Yes                                 | Yes                  |
| Main strength            | Simple segmentation | Understand hierarchy                | Density + noise      |

---

# 50. How do we decide which clustering algorithm to use?

This is an important interview scenario.

Suppose the interviewer gives you a new dataset and says:

> "Which clustering algorithm would you use?"

Don't immediately answer K-Means.

First understand the dataset.

```text
Dataset
 ↓
Understand features
 ↓
Preprocess
 ↓
Scale
 ↓
Explore data
 ↓
Try suitable algorithms
 ↓
Evaluate
 ↓
Compare
 ↓
Choose appropriate model
```

---

# 51. When would you choose K-Means?

Choose K-Means when:

```text
Need clear segmentation
+
Clusters are reasonably compact
+
Need to assign future/new data
```

Example:

> Customer segmentation where new customers continuously arrive and need to be assigned to an existing segment.

---

# 52. When would you choose Hierarchical?

Choose Hierarchical when:

```text
Need to understand relationships
+
Want hierarchical structure
+
Want dendrogram
+
Exploratory clustering is important
```

Example:

> Understanding how customer groups progressively merge at different similarity levels.

---

# 53. When would you choose DBSCAN?

Choose DBSCAN when:

```text
Number of clusters unknown
+
Density structure is meaningful
+
Irregular shapes possible
+
Noise/outliers matter
```

Example:

> Detecting dense geographic regions while identifying isolated points as noise.

---

# 54. Can we use all three algorithms on the same dataset?

**Yes.**

This is a very good practical approach when you don't know which clustering structure fits the dataset.

```text
Same dataset
     ↓
K-Means
     ↓
Hierarchical
     ↓
DBSCAN
     ↓
Compare results
```

But don't choose solely because one metric has the biggest/smallest number.

We consider:

```text
Metrics
+
Cluster structure
+
Noise
+
Visualization
+
Domain/business meaning
```

---

# 55. What clustering metrics do we use?

The main ones we studied:

```text
Silhouette Score
Davies-Bouldin Score
Calinski-Harabasz Score
```

---

# 56. Silhouette Score

Silhouette measures how well a point fits within its own cluster compared with other clusters.

Range:

```text
-1 to +1
```

Generally:

```text
Higher → better-separated/cohesive clusters
```

A value closer to 1 is generally desirable.

---

# 57. Davies-Bouldin Score

It evaluates cluster compactness and separation.

Generally:

```text
Lower → better
```

A lower value indicates clusters tend to be more compact and better separated.

---

# 58. Calinski-Harabasz Score

It compares between-cluster separation with within-cluster dispersion.

Generally:

```text
Higher → better
```

---

# 59. Which metric is best?

There is **no universal single best metric**.

A good answer is:

> "I would use multiple clustering metrics together and also consider the actual structure and business meaning of the clusters."

---

# 60. Why can't we use normal accuracy for clustering?

Because clustering normally has no known target labels.

Supervised:

```text
Actual y
   ↓
Prediction
   ↓
Compare
   ↓
Accuracy
```

Unsupervised:

```text
X only
 ↓
Clusters discovered
 ↓
No known target cluster
```

Therefore, standard classification accuracy is generally not applicable.

---

# 61. Complete K-Means workflow

```text
Dataset
 ↓
Select X
 ↓
Preprocess
 ↓
Scale
 ↓
Try different K
 ↓
Elbow / Silhouette
 ↓
Select K
 ↓
KMeans
 ↓
fit()
 ↓
Clusters
 ↓
Metrics
 ↓
Interpret
 ↓
New data
 ↓
transform()
 ↓
predict()
```

---

# 62. Complete Hierarchical workflow

```text
Dataset
 ↓
Select X
 ↓
Preprocess
 ↓
Scale
 ↓
Choose distance/linkage
 ↓
Create linkage matrix
 ↓
Dendrogram
 ↓
Choose number of clusters
 ↓
AgglomerativeClustering
 ↓
fit_predict()
 ↓
Clusters
 ↓
Metrics
 ↓
Interpret
```

---

# 63. Complete DBSCAN workflow

```text
Dataset
 ↓
Select X
 ↓
Preprocess
 ↓
Scale
 ↓
Starting min_samples
 ↓
K-distance plot
 ↓
Candidate eps
 ↓
Test eps + min_samples
 ↓
DBSCAN.fit_predict()
 ↓
Clusters + Noise
 ↓
Metrics + noise analysis
 ↓
Compare parameter combinations
 ↓
Final DBSCAN
 ↓
Interpret clusters
```

---

# 64. Why do we scale before all three?

Because the algorithms we are discussing rely on distances or distance-related calculations.

```text
Raw data
 ↓
Different feature scales
 ↓
Distance can be dominated by large-scale features
```

So:

```python
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

Then use:

```python
X_scaled
```

for clustering.

---

# 65. What is `fit()` vs `fit_predict()` in clustering?

### `fit()`

Learns the clustering structure/model.

Example:

```python
model.fit(X_scaled)
```

### `fit_predict()`

Fits the model and returns cluster labels.

```python
labels = model.fit_predict(X_scaled)
```

For clustering algorithms where labels are produced directly from the clustering operation, `fit_predict()` is convenient.

---

# 66. Why do we not manually calculate everything?

Because the algorithm implementation already contains the mathematical procedure.

For example, in K-Means:

```python
model.fit(X_scaled)
```

internally handles:

```text
Centroid initialization
Distance calculation
Assignment
Centroid update
Iteration
Convergence
```

Similarly, DBSCAN internally handles:

```text
Neighborhood search
Core points
Density connectivity
Cluster expansion
Noise
```

And Hierarchical clustering handles:

```text
Distance
Cluster merging
Linkage
Hierarchy
```

We understand the mathematics conceptually, but we don't need to manually implement it for normal project usage.

---

# 67. What is the biggest conceptual difference among the three?

Remember this:

```text
K-Means
"Which centroid is this point closest to?"

Hierarchical
"How are these points/groups related as distance increases?"

DBSCAN
"Where are the dense regions, and which points are isolated?"
```

If you remember these three questions, you can explain the fundamental difference.

---

# 68. Scenario-based interview question

### Interviewer:

> "You have customer data, but you don't know how many customer segments exist. What will you do?"

### Answer:

> "Since this is an unsupervised problem, I would first preprocess and scale the relevant features. I would explore the clustering structure using appropriate algorithms. With K-Means, I could evaluate different K values using the Elbow Method and Silhouette Score. I could also use Hierarchical Clustering and inspect the dendrogram, and DBSCAN if density and noise are relevant. I would compare the resulting cluster quality and business interpretability before selecting the final approach."

---

# 69. Scenario: New customers arrive every day

### Which model is more convenient?

**K-Means** is generally more convenient when you need to assign new observations to learned clusters:

```text
Existing data
 ↓
K-Means
 ↓
Centroids
 ↓
New customer
 ↓
predict()
```

---

# 70. Scenario: You have lots of outliers

Which algorithm may be useful?

**DBSCAN**, because noise points can be identified as:

```text
-1
```

But the suitability still depends on the actual density structure of the data.

---

# 71. Scenario: You want to see how groups are related

Which algorithm?

**Hierarchical Clustering.**

Because:

```text
Data
 ↓
Merging hierarchy
 ↓
Dendrogram
```

shows the structure at different distance levels.

---

# 72. Scenario: Clusters have strange/irregular shapes

Which algorithm may be useful?

**DBSCAN** can be a strong candidate because it is density-based and can identify non-spherical cluster structures.

---

# 73. Scenario: You need simple customer segmentation

Which algorithm?

**K-Means** is often a good starting point when clusters are reasonably compact and you want to assign future customers.

---

# 74. What if metrics disagree?

Example:

```text
Model A
Silhouette → better
DB → worse

Model B
Silhouette → worse
DB → better
```

Don't automatically choose one metric.

Answer:

> "I would consider multiple metrics, inspect the cluster structure, examine noise/outliers where relevant, and validate whether the clusters are meaningful for the business problem."

---

# 75. What is the difference between clustering and classification?

### Classification

We already know the classes.

```text
Training data
X + y
 ↓
Learn classes
 ↓
Predict class
```

### Clustering

We don't know the groups.

```text
X
 ↓
Discover groups
```

Example:

```text
Classification:
Customer → Premium / Regular

Clustering:
Find customer groups automatically
```

---

# 76. What is the difference between regression and clustering?

### Regression

Predicts a continuous value.

```text
House features
 ↓
Price
```

### Clustering

Discovers groups.

```text
Customer features
 ↓
Customer groups
```

---

# 77. Is clustering always the final answer?

No.

Clustering gives us groups, but we still need to understand:

```text
What does Cluster 0 mean?
What does Cluster 1 mean?
What does Cluster 2 mean?
```

For example:

```text
Cluster 0
→ High income + high spending

Cluster 1
→ Low income + low spending

Cluster 2
→ High income + low spending
```

This is called **cluster interpretation/profiling**.

---

# 78. What is cluster profiling?

After clustering:

```python
df["Cluster"] = labels
```

we can calculate characteristics:

```python
df.groupby("Cluster")[
    [
        "Age",
        "Annual_Income",
        "Spending_Score"
    ]
].mean()
```

This helps us understand what each cluster represents.

---

# 79. Important interview trap

### Question:

> "The Silhouette Score is 0.85, so is the model 85% accurate?"

### Correct answer:

**No.**

Silhouette Score is **not accuracy**.

It measures the quality/separation of clustering.

---

# 80. Important interview trap

### Question:

> "K-Means has the lowest inertia, so it is definitely the best model."

### Correct answer:

**No.**

Inertia generally decreases as K increases.

We need to consider:

```text
Elbow
+
Silhouette
+
Cluster structure
+
Business meaning
```

---

# 81. Important interview trap

### Question:

> "DBSCAN has no predict(), so it isn't useful."

### Correct answer:

**Wrong.**

DBSCAN is useful for discovering dense regions and identifying noise/outliers. Its primary purpose is not necessarily assigning arbitrary future points to pre-existing clusters.

---

# 82. Important interview trap

### Question:

> "Hierarchical clustering only tells us the distance."

### Correct answer:

Not exactly.

It uses distances and linkage to **build a hierarchy of clusters**. The dendrogram shows how clusters merge at different distance levels.

---

# 83. Important interview trap

### Question:

> "Do we manually identify every core and border point in DBSCAN?"

### Correct answer:

No.

We understand the logic conceptually, but the algorithm performs the neighborhood search, core/border/noise identification, and cluster expansion internally.

---

# 84. One complete project explanation

If the interviewer asks:

> **"Explain your unsupervised learning project."**

You can say:

> "I worked on customer clustering using unsupervised learning. Since the dataset did not contain a target variable, I treated it as an unsupervised problem. I selected relevant customer features such as Age, Annual Income and Spending Score and standardized them because the clustering algorithms rely on distances. I explored K-Means, Hierarchical Clustering and DBSCAN because each uses a different clustering strategy. For K-Means, I evaluated different values of K using the Elbow Method and Silhouette Score. For Hierarchical Clustering, I used a linkage method and a dendrogram to understand the hierarchical structure and select the number of clusters. For DBSCAN, I selected a starting `min_samples`, used a K-distance plot to estimate `eps`, and evaluated different parameter combinations. Finally, I compared the clustering quality using Silhouette, Davies-Bouldin and Calinski-Harabasz scores, along with the actual cluster structure and business interpretation."

---

# 85. Your complete mental map

Memorize this:

```text
                 UNSUPERVISED LEARNING
                          │
             ┌────────────┴────────────┐
             │                         │
         Clustering          Dimensionality Reduction
             │                         │
       ┌─────┼─────┐                  PCA
       │     │     │
   K-Means  Hier.  DBSCAN
       │     │     │
   Centroid Distance Density
       │     │     │
       │     │     ├── eps
       │     │     └── min_samples
       │     │
       │     └── Dendrogram
       │
       └── K
```

### K-Means

```text
Choose K
 ↓
Centroids
 ↓
Distance
 ↓
Assignment
 ↓
Update centroid
 ↓
Repeat
 ↓
Clusters
 ↓
predict() for new data
```

### Hierarchical

```text
Individual points
 ↓
Distance
 ↓
Merge closest
 ↓
Linkage
 ↓
Hierarchy
 ↓
Dendrogram
 ↓
Choose clusters
```

### DBSCAN

```text
eps + min_samples
 ↓
Density
 ↓
Core
 ↓
Border
 ↓
Noise
 ↓
Density-connected clusters
```

### Evaluation

```text
Clustering result
 ↓
Silhouette       ↑
Davies-Bouldin   ↓
Calinski-Harabasz ↑
Noise analysis
 ↓
Business meaning
```

---

# 86. Final interview cheat sheet

If you remember only this before an interview:

| Question                           | Answer                                                                  |
| ---------------------------------- | ----------------------------------------------------------------------- |
| What is unsupervised learning?     | Learning patterns from unlabeled data                                   |
| Main clustering algorithms?        | K-Means, Hierarchical, DBSCAN                                           |
| K-Means idea?                      | Centroid-based                                                          |
| Hierarchical idea?                 | Distance + merging                                                      |
| DBSCAN idea?                       | Density-based                                                           |
| K-Means needs K?                   | Yes                                                                     |
| DBSCAN needs K?                    | No                                                                      |
| Hierarchical uses?                 | Dendrogram/hierarchy                                                    |
| DBSCAN parameters?                 | `eps`, `min_samples`                                                    |
| K-Means new data?                  | `predict()`                                                             |
| Hierarchical new-data `predict()`? | Standard sklearn: No                                                    |
| DBSCAN new-data `predict()`?       | Standard sklearn: No                                                    |
| K-Means handles noise well?        | No                                                                      |
| DBSCAN detects noise?              | Yes                                                                     |
| K-Means best for?                  | Compact segmentation + new-data assignment                              |
| Hierarchical best for?             | Understanding relationships/hierarchy                                   |
| DBSCAN best for?                   | Dense/irregular groups + noise                                          |
| Silhouette?                        | Higher generally better                                                 |
| Davies-Bouldin?                    | Lower generally better                                                  |
| Calinski-Harabasz?                 | Higher generally better                                                 |
| Clustering accuracy?               | Not normally standard accuracy                                          |
| Why scaling?                       | Distance-based algorithms are affected by feature scale                 |
| How choose K?                      | Elbow + Silhouette + domain meaning                                     |
| How choose DBSCAN eps?             | K-distance plot + tuning                                                |
| How choose DBSCAN min_samples?     | Reasonable starting value + tuning                                      |
| Why compare models?                | Different algorithms make different assumptions about cluster structure |

**The one sentence that separates all three:**

> **K-Means asks "which centroid is closest?", Hierarchical asks "how do groups progressively merge?", and DBSCAN asks "where are the dense regions and which points are noise?"**

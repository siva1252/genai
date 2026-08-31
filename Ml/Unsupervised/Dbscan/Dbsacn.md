k-distance plot
       ↓
find elbow
       ↓
candidate eps

this is the way of eps

Check how many points are inside eps
              ↓
        count >= min_samples?
          /             \
        YES              NO
         ↓                ↓
       CORE        Is it within eps
                   of a CORE point?
                    /        \
                  YES         NO
                   ↓           ↓
                BORDER       NOISE



# DBSCAN — Interview Questions & Answers

## 1. What is DBSCAN?

**Answer:**

DBSCAN stands for **Density-Based Spatial Clustering of Applications with Noise**.

It is an **unsupervised clustering algorithm** that groups data points based on **density** and can identify **noise/outliers**.

```text
DBSCAN
  ↓
Density
  ↓
Clusters + Noise
```

---

## 2. Why do we use DBSCAN?

**Answer:**

We use DBSCAN when:

* We don't know the number of clusters beforehand.
* We want to find clusters based on density.
* The clusters may have irregular shapes.
* We want to identify noise/outliers.

For example:

```text
Customer data
      ↓
Dense customer groups
      ↓
Clusters

Isolated customers
      ↓
Noise
```

---

## 3. What is the main difference between K-Means and DBSCAN?

| K-Means                                   | DBSCAN                                   |
| ----------------------------------------- | ---------------------------------------- |
| Centroid-based                            | Density-based                            |
| Need to specify `n_clusters`              | Don't specify number of clusters         |
| Sensitive to outliers                     | Can identify noise                       |
| Works well for roughly spherical clusters | Can find irregular-shaped clusters       |
| Has `predict()`                           | sklearn DBSCAN has no normal `predict()` |

Simple interview answer:

> **K-Means forms clusters around centroids, whereas DBSCAN forms clusters based on density and can identify noise points.**

---

# 4. What are the main parameters of DBSCAN?

There are two important parameters:

```python
DBSCAN(
    eps=0.38,
    min_samples=6
)
```

### `eps`

Defines the **maximum neighborhood distance** around a point.

### `min_samples`

Defines the **minimum number of samples required within that neighborhood** for a point to qualify as a core point.

---

# 5. What is `eps`?

**Answer:**

`eps` defines the radius around each data point.

For example:

```text
eps = 0.38
```

means DBSCAN checks which points are within that distance from a given point.

```text
        ●
    ●   ●   ●
        A
    ●       ●

←── eps ──→
```

The actual distance is calculated in the feature space, so scaling is usually important.

---

# 6. What is `min_samples`?

**Answer:**

`min_samples` determines how many samples must be present in the `eps` neighborhood for a point to be considered a **core point**.

Example:

```text
eps = 0.38
min_samples = 6
```

If the neighborhood satisfies the required sample count:

```text
→ Core point
```

Otherwise, the point is not a core point.

---

# 7. How do you choose `eps`?

**Answer:**

A common approach is to use a **K-distance plot**.

The process is:

```text
Choose min_samples
       ↓
Find k-nearest-neighbor distances
       ↓
Sort distances
       ↓
Plot them
       ↓
Find elbow/bend
       ↓
Use that region as candidate eps
```

Example:

```python
from sklearn.neighbors import NearestNeighbors

neighbors = NearestNeighbors(
    n_neighbors=min_samples
)

neighbors.fit(x_scaled)

distances, indices = neighbors.kneighbors(
    x_scaled
)

k_distances = sorted(
    distances[:, -1]
)
```

Then plot the distances and inspect the elbow.

---

# 8. How do you choose `min_samples`?

**Answer:**

There is no single value that is correct for every dataset.

We can start with a reasonable value based on the number of features and then tune it.

For example, with 3 features:

```text
Age
Annual Income
Spending Score
```

we might start with:

```python
min_samples = 6
```

Then test values such as:

```text
4, 5, 6, 7, 8
```

and evaluate the resulting clustering.

**Important interview point:**

> `min_samples = 6` is not a universal rule. It is a starting point that should be tuned according to the dataset.

---

# 9. What are Core, Border and Noise points?

DBSCAN classifies points into three concepts:

### Core Point

A point that has enough samples in its `eps` neighborhood.

```text
        ●
    ●   A   ●
        ●
      ● ●

A → Core
```

### Border Point

A point that does not itself satisfy the core requirement but lies within the `eps` neighborhood of a core point.

```text
● ● ●
 ● A ●     X

A → Core
X → Border
```

### Noise Point

A point that is neither a core point nor density-connected to a core point.

```text
● ● ●


              X

X → Noise
```

In scikit-learn:

```text
Core   → internally identified by DBSCAN
Border → internally identified by DBSCAN
Noise  → label -1
```

---

# 10. How does DBSCAN actually create clusters?

**Answer:**

Suppose:

```text
eps = 0.38
min_samples = 6
```

DBSCAN examines the points.

```text
Point
  ↓
Find points inside eps
  ↓
Count them
  ↓
Enough samples?
  ↓
Core point
  ↓
Expand cluster
  ↓
Find connected core points
  ↓
Add reachable border points
  ↓
Unconnected points become noise
```

So DBSCAN automatically discovers the clusters.

We don't manually write:

```text
Customer 1 → Core
Customer 2 → Border
Customer 3 → Noise
```

The algorithm does this internally.

---

# 11. What does "connected" mean in DBSCAN?

This is an important interview question.

Suppose:

```text
A → B → C
```

and the points are connected through density-reachable core points.

DBSCAN can place them in the same cluster even if A and C are not directly close to each other.

So DBSCAN is based on **density connectivity**, not simply:

> "Every point must be close to every other point."

---

# 12. How does DBSCAN identify noise?

**Answer:**

If a point:

1. Does not satisfy the core-point requirement, and
2. Is not reachable from a core point,

then DBSCAN marks it as noise.

In scikit-learn:

```python
label = -1
```

means noise.

---

# 13. Does DBSCAN require us to specify the number of clusters?

**Answer:**

No.

With K-Means:

```python
KMeans(n_clusters=3)
```

we specify the number of clusters.

With DBSCAN:

```python
DBSCAN(
    eps=0.38,
    min_samples=6
)
```

we don't specify `n_clusters`.

DBSCAN discovers the number of clusters based on density.

---

# 14. Why do we scale data before DBSCAN?

**Answer:**

DBSCAN uses **distance** to determine neighborhoods.

Suppose:

```text
Age              → 20–60
Annual Income    → 20,000–150,000
Spending Score   → 1–100
```

Income has a much larger numerical scale.

Without scaling, income can dominate the distance calculation.

Therefore:

```python
scaler = StandardScaler()

x_scaled = scaler.fit_transform(x)
```

Then we apply DBSCAN to:

```python
x_scaled
```

---

# 15. What happens inside `model.fit_predict()`?

```python
model = DBSCAN(
    eps=0.38,
    min_samples=6
)

labels = model.fit_predict(x_scaled)
```

Internally, DBSCAN:

```text
Every data point
      ↓
Check eps neighborhood
      ↓
Count samples
      ↓
Identify core points
      ↓
Expand density-connected clusters
      ↓
Identify border points
      ↓
Identify noise
      ↓
Return cluster labels
```

Example output:

```text
0
0
1
1
0
-1
2
2
```

Here:

```text
0, 1, 2 → clusters
-1      → noise
```

---

# 16. How do you evaluate DBSCAN?

We can use clustering metrics such as:

### Silhouette Score

**Higher is better.**

Measures how well-separated and cohesive the clusters are.

### Davies-Bouldin Score

**Lower is better.**

Lower means clusters are more compact and better separated.

### Calinski-Harabasz Score

**Higher is generally better.**

Measures between-cluster separation relative to within-cluster dispersion.

We also check:

```text
Noise count / noise percentage
```

because DBSCAN can mark points as noise.

---

# 17. Why do we remove noise before calculating these metrics?

Suppose:

```text
labels = [0, 0, 1, 1, 2, -1]
```

Here:

```text
-1 = noise
```

For many standard clustering metrics, we evaluate the actual clusters separately from the noise points.

```python
mask = labels != -1

x_clustered = x_scaled[mask]

labels_clustered = labels[mask]
```

Then calculate:

```python
silhouette_score(
    x_clustered,
    labels_clustered
)
```

---

# 18. How do you find the best DBSCAN parameters?

**Answer:**

We don't blindly choose one `eps` and `min_samples`.

We can test combinations:

```text
eps          min_samples

0.35             4
0.35             5
0.35             6

0.38             4
0.38             5
0.38             6

0.40             4
0.40             5
0.40             6
```

For every combination:

```text
DBSCAN
 ↓
Clusters + Noise
 ↓
Metrics
 ↓
Compare
```

Then choose parameters that give good cluster quality **and** a reasonable amount of noise.

---

# 19. What is the complete DBSCAN workflow?

**Interview answer:**

> First, I load the dataset and select relevant features. Then I preprocess and scale the features because DBSCAN is distance-based. I choose a reasonable starting `min_samples` and use a K-distance plot to estimate a candidate `eps`. Then I test different `eps` and `min_samples` combinations using DBSCAN. I evaluate the resulting clusters using Silhouette, Davies-Bouldin and Calinski-Harabasz scores, along with the amount of noise. Finally, I select suitable parameters, train the final DBSCAN model and interpret the clusters.

---

# 20. Does DBSCAN have `predict()`?

**Answer:**

Standard scikit-learn `DBSCAN` does **not provide a normal `predict()` method for assigning arbitrary new points**.

```python
model.fit_predict(X)
```

is used on the data being clustered.

This is different from K-Means:

```python
kmeans.fit(X)

kmeans.predict(new_data)
```

---

# 21. Then how is DBSCAN useful if it doesn't predict new data?

**Answer:**

DBSCAN is useful when the main requirement is:

```text
Existing data
      ↓
Discover dense groups
      +
Identify unusual/noise points
```

For example:

* anomaly/outlier discovery
* geographic/spatial clustering
* customer behavior with irregular groups
* datasets where the number of clusters is unknown
* datasets containing noise

Its value is **discovering the structure of the existing data**, not necessarily assigning every future observation.

---

# 22. What is the main advantage of DBSCAN?

**Answer:**

The major advantages are:

1. **No need to specify number of clusters.**
2. Can find **irregularly shaped clusters**.
3. Can identify **noise/outliers**.
4. It is density-based.

---

# 23. What are the disadvantages of DBSCAN?

**Answer:**

Main limitations:

1. Choosing good `eps` and `min_samples` can be difficult.
2. It can struggle when different clusters have **very different densities**.
3. Distance becomes problematic in very high-dimensional data.
4. Standard sklearn DBSCAN doesn't provide a normal `predict()` for new data.

---

# 24. DBSCAN vs Hierarchical Clustering

| DBSCAN                    | Hierarchical                                               |
| ------------------------- | ---------------------------------------------------------- |
| Density-based             | Distance/hierarchy-based                                   |
| Uses `eps`, `min_samples` | Uses linkage and distance                                  |
| Finds dense regions       | Builds hierarchy of clusters                               |
| Identifies noise          | Doesn't naturally provide DBSCAN-style noise labels        |
| No normal `predict()`     | Standard AgglomerativeClustering has no normal `predict()` |
| No `n_clusters` required  | Can choose `n_clusters`                                    |

---

# 25. DBSCAN vs K-Means

### Interview answer:

> K-Means is centroid-based and requires the number of clusters to be specified. DBSCAN is density-based, doesn't require the number of clusters beforehand, can detect noise, and can discover non-spherical cluster shapes.

```text
K-Means
→ Centroid
→ Cluster
→ New-data assignment

DBSCAN
→ Density
→ Cluster
→ Noise
```

---

# 26. What does `eps` control?

**Answer:**

It controls the **size of the neighborhood** around each point.

```text
Small eps
→ smaller neighborhoods
→ potentially more noise/smaller clusters

Large eps
→ larger neighborhoods
→ potentially fewer/larger clusters
```

The exact effect depends on the dataset.

---

# 27. What happens if `eps` is too small?

**Answer:**

Very few points may satisfy the density requirement.

Possible result:

```text
Many noise points
+
Small clusters
```

---

# 28. What happens if `eps` is too large?

**Answer:**

Too many points can become connected.

Possible result:

```text
Different groups merge
→ fewer/larger clusters
```

---

# 29. What happens if `min_samples` is too high?

**Answer:**

The density requirement becomes stricter.

Possible result:

```text
Fewer core points
→ more noise
→ fewer clusters
```

---

# 30. What happens if `min_samples` is too low?

**Answer:**

The density requirement becomes easier to satisfy.

Possible result:

```text
More core points
→ more points become connected
→ clusters may merge
→ noise may decrease
```

---

# 31. Why can't we say DBSCAN accuracy is 95%?

**Answer:**

Because DBSCAN is **unsupervised**.

There is normally no target `y` containing the correct cluster for every observation.

Therefore, we don't normally calculate:

```text
Accuracy = correct predictions / total predictions
```

Instead, we evaluate cluster quality using:

```text
Silhouette
Davies-Bouldin
Calinski-Harabasz
Noise analysis
Business/domain interpretation
```

---

# 32. Explain your DBSCAN project in an interview

You can say:

> "I used DBSCAN for customer clustering. First, I selected Age, Annual Income and Spending Score as features and standardized them using StandardScaler because DBSCAN is distance-based. I selected a starting `min_samples` and used a K-distance plot to identify a suitable range for `eps`. I then applied DBSCAN using different parameter combinations and evaluated the resulting clusters using Silhouette, Davies-Bouldin and Calinski-Harabasz scores, while also checking the number of noise points. Finally, I selected suitable parameters, fitted the final DBSCAN model and analyzed the characteristics of each cluster."

That is a **strong interview-level explanation of your complete DBSCAN implementation**.

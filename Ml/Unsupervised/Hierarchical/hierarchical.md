The main output is a hierarchical tree called a dendrogram.


Hierarchical Clustering
       ↓
├── Agglomerative
└── Divisive


Agglomerative

Bottom → Up

Start with every point as its own cluster and repeatedly merge the closest clusters.

Divisive

Top → Down

Start with all points in one cluster and repeatedly split them



All customers
     ↓
Each customer starts as separate cluster
     ↓
Find closest customers/clusters
     ↓
Merge them
     ↓
Again find closest clusters
     ↓
Merge again
     ↓
Continue until hierarchy is created
     ↓
Dendrogram


Which customers are similar?
Which groups naturally exist?
Which customers are closely related?
At what level do these groups separate?
Hierarchical Clustering is very useful for this.



# Hierarchical Clustering — Complete Interview Notes

## 1. What is Hierarchical Clustering?

**Hierarchical Clustering is an unsupervised learning algorithm used to find natural groups in unlabeled data by building a hierarchy of clusters based on similarity/distance.**

Example:

```text
Customer Dataset
       ↓
No target / label
       ↓
Find similar customers
       ↓
Create customer groups
```

---

# 2. When do we choose Hierarchical Clustering?

First identify the problem:

```text
Dataset
   ↓
Do we have target y?
   ↓
NO
   ↓
Unsupervised Learning
   ↓
What is the requirement?
   ↓
Find groups
   ↓
Clustering
```

Then we can consider:

```text
K-Means
Hierarchical Clustering
DBSCAN
```

### When Hierarchical is useful

Use it when we want to:

* discover natural groups
* understand relationships between observations
* see how groups are formed at different levels
* visualize the hierarchy using a dendrogram

---

# 3. Example Dataset

Suppose we have customer data:

| Customer | Age | Annual Income | Spending Score |
| -------- | --: | ------------: | -------------: |
| A        |  22 |           25k |             80 |
| B        |  24 |           27k |             85 |
| C        |  25 |           30k |             78 |
| D        |  45 |           70k |             40 |
| E        |  48 |           72k |             35 |
| F        |  50 |           75k |             30 |
| G        |  60 |           90k |             15 |
| H        |  62 |           92k |             10 |

We don't have a target.

Our requirement is:

> Find groups of similar customers.

Therefore:

```text
No target
   ↓
Unsupervised
   ↓
Need groups
   ↓
Clustering
   ↓
Hierarchical Clustering
```

---

# 4. Types of Hierarchical Clustering

There are two main types:

```text
Hierarchical Clustering
        ↓
 ┌──────┴──────┐
 ↓             ↓
Agglomerative  Divisive
```

### Agglomerative

**Bottom → Up**

Start with every data point as its own cluster and repeatedly merge the closest clusters.

### Divisive

**Top → Down**

Start with all observations in one cluster and repeatedly split them.

For our practical implementation, we mainly use:

**Agglomerative Hierarchical Clustering.**

---

# 5. How Agglomerative Clustering Works

This is the most important part.

Initially every customer is a separate cluster:

```text
[A] [B] [C] [D] [E] [F] [G] [H]
```

So:

```text
8 customers
=
8 initial clusters
```

---

## Step 1 — Calculate distances

The algorithm determines how close customers/clusters are.

For example:

```text
A ↔ B → very close
A ↔ C → very close

D ↔ E → very close
E ↔ F → very close

G ↔ H → very close
```

---

## Step 2 — Merge the closest clusters

Suppose A and B are closest:

```text
[A B] [C] [D] [E] [F] [G] [H]
```

Now there are:

```text
7 clusters
```

---

## Step 3 — Again find the closest clusters

Suppose `[A B]` and C are closest:

```text
[A B C] [D] [E] [F] [G] [H]
```

Now:

```text
6 clusters
```

---

## Step 4 — Continue merging

For example:

```text
[A B C] [D E] [F] [G] [H]
```

Then:

```text
[A B C] [D E F] [G] [H]
```

Then:

```text
[A B C] [D E F] [G H]
```

Eventually everything can be merged into one hierarchy.

```text
8 → 7 → 6 → 5 → 4 → 3 → 2 → 1
```

This is the fundamental process.

---

# 6. Does Hierarchical use centroids?

**No, not like K-Means.**

### K-Means

```text
Centroids
   ↓
Calculate distance to centroids
   ↓
Assign points
   ↓
Recalculate centroids
   ↓
Repeat
```

### Hierarchical

```text
Clusters
   ↓
Calculate distance between clusters
   ↓
Merge closest clusters
   ↓
Repeat
```

So don't say:

> "Hierarchical repeatedly recalculates centroids."

That's K-Means.

---

# 7. What is Linkage?

Once we have clusters, we need a rule to determine:

> **How do we calculate the distance between two clusters?**

That rule is called **linkage**.

Common methods:

```text
Single
Complete
Average
Ward
```

---

# 8. Single Linkage

Uses the **minimum distance** between points in two clusters.

```text
Cluster A          Cluster B

● ---------------- ●
      closest pair
```

It asks:

> What is the closest pair of observations between these clusters?

### Problem

It can produce **chaining**.

---

# 9. Complete Linkage

Uses the **maximum distance** between observations in the two clusters.

```text
Cluster A                  Cluster B

● --------------------------- ●
          farthest pair
```

It generally produces more compact clusters than single linkage.

---

# 10. Average Linkage

Uses the **average pairwise distance** between observations in the two clusters.

```text
All cross-cluster distances
          ↓
Calculate average
          ↓
Cluster distance
```

---

# 11. Ward Linkage

Ward linkage merges clusters in a way that minimizes the increase in **within-cluster variance**.

It generally produces compact clusters.

In our code:

```python
linkage(X_scaled, method="ward")
```

we are using Ward linkage.

---

# 12. Why do we scale the data?

Our features have different scales:

```text
Age              → around 20–60
Income           → thousands
Spending Score   → around 0–100
```

Because Hierarchical Clustering is distance-based, a large-scale feature can dominate the distance.

Therefore:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

### Interview answer

> "I scale the features because hierarchical clustering relies on distance calculations, and features with larger numerical scales can dominate the distance."

---

# 13. What is a Dendrogram?

A **dendrogram is a tree-like diagram that represents the hierarchy of cluster merging.**

It shows:

* which clusters were merged
* the order of merging
* the distance at which they were merged

Conceptually:

```text
Distance
   |
   |              ┌───────────────┐
   |          ┌───┤               │
   |      ┌───┤   │               │
   |   ┌──┴─┐ │   │            ┌──┴──┐
   |   A    B C   D            E     F
   |
   +------------------------------------
```

The **height** of a merge represents its distance.

---

# 14. How do we find the number of clusters?

This is an important difference from K-Means.

In Hierarchical Clustering:

```text
Dataset
   ↓
Build hierarchy
   ↓
Dendrogram
   ↓
Draw horizontal cut
   ↓
Count branches
   ↓
Number of clusters
```

For example:

```text
Horizontal cut
-------------------------
      ↓       ↓       ↓

   Cluster 1  Cluster 2  Cluster 3
```

If the horizontal line crosses **3 main branches**:

```text
n_clusters = 3
```

---

# 15. Why is the dendrogram important?

It lets us see the hierarchy before deciding the final cluster level.

For example:

```text
Distance
   |
   |       ┌───────────────┐
   |       │               │
   |   ┌───┴───┐       ┌───┴───┐
   |   │       │       │       │
   |   A       B       C       D
```

If we cut lower:

```text
A | B | C | D
```

we get more clusters.

If we cut higher:

```text
A B | C D
```

we get fewer clusters.

Therefore:

> **The cut level determines the number of final clusters.**

---

# 16. Coding Flow

## Import libraries

```python
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

from scipy.cluster.hierarchy import dendrogram, linkage

import matplotlib.pyplot as plt

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
```

---

# 17. Load Dataset

```python
df = pd.read_csv("../data/customer_clustering.csv")
```

---

# 18. Select Features

```python
X = df[
    [
        "Age",
        "Annual_Income",
        "Spending_Score"
    ]
]
```

There is no:

```python
y
```

because this is unsupervised learning.

---

# 19. Scale Features

```python
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

---

# 20. Build the Hierarchy

```python
Z = linkage(
    X_scaled,
    method="ward"
)
```

Here:

```text
linkage()
   ↓
Builds the hierarchical merging structure
```

Ward determines how clusters are merged based on within-cluster variance.

---

# 21. Display Dendrogram

```python
plt.figure(figsize=(10, 7))

dendrogram(Z)

plt.title("Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Distance")

plt.show()
```

Now inspect the dendrogram and determine a suitable cluster configuration.

---

# 22. Create the Model

Suppose we decide to test:

```text
3 clusters
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

# 23. Fit and Get Cluster Labels

```python
labels = model.fit_predict(X_scaled)
```

This gives something like:

```text
Customer A → Cluster 0
Customer B → Cluster 0
Customer C → Cluster 0

Customer D → Cluster 1
Customer E → Cluster 1
Customer F → Cluster 1

Customer G → Cluster 2
Customer H → Cluster 2
```

We can store them:

```python
df["Cluster"] = labels
```

---

# 24. Evaluate the Clusters

Now we need to determine whether our clusters are good.

### Silhouette Score

```python
silhouette = silhouette_score(
    X_scaled,
    labels
)
```

**Higher is better.**

---

### Davies-Bouldin Index

```python
davies = davies_bouldin_score(
    X_scaled,
    labels
)
```

**Lower is better.**

---

### Calinski-Harabasz Score

```python
calinski = calinski_harabasz_score(
    X_scaled,
    labels
)
```

**Higher is better.**

---

# 25. Metric Summary

| Metric                  | Better       |
| ----------------------- | ------------ |
| Silhouette Score        | **Higher ↑** |
| Davies-Bouldin Index    | **Lower ↓**  |
| Calinski-Harabasz Score | **Higher ↑** |

Remember:

```text
Silhouette       ↑
Davies-Bouldin   ↓
Calinski         ↑
```

---

# 26. Why do we compare metrics?

Suppose we test:

```text
K-Means
Hierarchical
DBSCAN
```

on the same dataset.

We want to know:

> Which algorithm creates the best-quality clusters for this dataset?

Example:

| Model        | Silhouette | Davies-Bouldin | Calinski |
| ------------ | ---------: | -------------: | -------: |
| K-Means      |       0.62 |           0.55 |      400 |
| Hierarchical |       0.71 |           0.42 |      520 |
| DBSCAN       |       0.48 |           0.70 |      300 |

Here Hierarchical looks better according to all three metrics.

But we don't choose only from metrics.

We also consider:

```text
Metrics
   +
Cluster structure
   +
Business meaning
   +
Outliers
   +
Future-data requirement
```

---

# 27. Very Important — New Data

This is one of the biggest differences between K-Means and standard Agglomerative Clustering.

### K-Means

K-Means learns centroids:

```text
Training
   ↓
Learn centroids
   ↓
New customer
   ↓
model.predict()
   ↓
Calculate distance to centroids
   ↓
Nearest centroid
   ↓
Cluster
```

So:

```python
model.predict(new_data)
```

is supported.

---

### Hierarchical

Standard sklearn `AgglomerativeClustering` does not provide the same direct:

```python
model.predict(new_data)
```

workflow for unseen data.

Why?

Because it learns a **hierarchy of cluster merges**, not reusable centroids.

So:

```text
Hierarchical
   ↓
Existing dataset
   ↓
Build hierarchy
   ↓
Dendrogram
   ↓
Final clusters
```

It is particularly useful for **exploring relationships and grouping structure in the existing dataset**.

---

# 28. Is Hierarchical useless for new data?

**No.**

It is not useless.

Its strength is different.

For example:

> "I have an existing customer dataset and I want to understand how customers naturally group together and how those groups are related."

Hierarchical Clustering is very useful.

But if the requirement is:

> "New customers arrive every day and I need to assign each one to an existing cluster."

K-Means is generally more convenient because it has learned centroids and provides a direct `predict()` workflow.

---

# 29. K-Means vs Hierarchical

| K-Means                              | Hierarchical                                              |
| ------------------------------------ | --------------------------------------------------------- |
| Centroid-based                       | Hierarchy-based                                           |
| K is central to training             | Builds hierarchy                                          |
| Recalculates centroids               | Merges clusters                                           |
| Uses centroid distance               | Uses cluster distance/linkage                             |
| No dendrogram                        | Dendrogram                                                |
| Usually efficient for large datasets | Can be computationally expensive                          |
| Direct `predict()` for new data      | Standard sklearn implementation has no direct `predict()` |
| Good for future cluster assignment   | Good for exploring hierarchical relationships             |

---

# 30. Complete Hierarchical Flow

```text
                    DATASET
                       ↓
                 Select X
                       ↓
                 Scale X
                       ↓
              Calculate hierarchy
                       ↓
                Linkage method
                       ↓
             Repeatedly merge
                       ↓
                  Dendrogram
                       ↓
             Choose horizontal cut
                       ↓
             Select cluster count
                       ↓
          AgglomerativeClustering
                       ↓
                 fit_predict()
                       ↓
               Cluster labels
                       ↓
             Evaluate clusters
                       ↓
      ┌────────────────┼────────────────┐
      ↓                ↓                ↓
 Silhouette       Davies-Bouldin    Calinski-Harabasz
      ↓                ↓                ↓
   Higher            Lower             Higher
                       ↓
              Interpret clusters
```

---

# 31. K-Means vs Hierarchical — Process

### K-Means

```text
Dataset
   ↓
Scale
   ↓
Try K values
   ↓
Elbow + Silhouette
   ↓
Choose K
   ↓
K centroids
   ↓
Assign points
   ↓
Recalculate centroids
   ↓
Repeat until convergence
   ↓
Final clusters
   ↓
Metrics
   ↓
New data → predict()
```

### Hierarchical

```text
Dataset
   ↓
Scale
   ↓
Start each point as separate cluster
   ↓
Calculate cluster distances
   ↓
Linkage
   ↓
Merge closest clusters
   ↓
Repeat
   ↓
Dendrogram
   ↓
Horizontal cut
   ↓
Final clusters
   ↓
Metrics
   ↓
No standard direct predict()
```

---

# 32. Interview Questions & Answers

### Q1. What is Hierarchical Clustering?

**Answer:**

> "Hierarchical Clustering is an unsupervised learning technique that builds a hierarchy of clusters based on distances between observations or clusters."

---

### Q2. What are the types?

**Answer:**

> "The two main types are Agglomerative and Divisive. Agglomerative works bottom-up by repeatedly merging clusters, while Divisive works top-down by repeatedly splitting clusters."

---

### Q3. Explain Agglomerative Clustering.

**Answer:**

> "It starts with each observation as an individual cluster. It calculates distances between clusters, merges the closest clusters according to the selected linkage method, and repeats this process to build a hierarchy."

---

### Q4. What is a dendrogram?

**Answer:**

> "A dendrogram is a tree-like visualization that represents the hierarchy of cluster merges and the distance at which those merges occur."

---

### Q5. How do you select the number of clusters?

**Answer:**

> "I inspect the dendrogram and choose an appropriate horizontal cut based on the separation between merges. I can then evaluate candidate cluster counts using metrics such as Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Score."

---

### Q6. What is linkage?

**Answer:**

> "Linkage defines how the distance between two clusters is calculated."

---

### Q7. What linkage methods do you know?

**Answer:**

> "Single, Complete, Average, and Ward linkage."

---

### Q8. Explain Single Linkage.

**Answer:**

> "Single linkage uses the minimum distance between observations from two clusters."

---

### Q9. Explain Complete Linkage.

**Answer:**

> "Complete linkage uses the maximum distance between observations from two clusters."

---

### Q10. Explain Average Linkage.

**Answer:**

> "Average linkage uses the average pairwise distance between observations from two clusters."

---

### Q11. Explain Ward Linkage.

**Answer:**

> "Ward linkage merges clusters while minimizing the increase in within-cluster variance, generally encouraging compact clusters."

---

### Q12. Does Hierarchical Clustering use centroids?

**Answer:**

> "Not like K-Means. Standard agglomerative hierarchical clustering works by calculating distances between clusters and merging them according to the linkage method."

---

### Q13. Why do we scale the data?

**Answer:**

> "Because the algorithm relies on distances. If features have very different scales, a large-scale feature can dominate the distance calculation."

---

### Q14. Does Hierarchical Clustering require K?

**Answer:**

> "The hierarchy can be constructed without first committing to the final number of clusters. We can inspect the dendrogram and choose a suitable cut. In sklearn's AgglomerativeClustering, we can then specify `n_clusters` for the desired final clustering."

---

### Q15. Does Hierarchical use the Elbow Method?

**Answer:**

> "The Elbow Method is mainly associated with K-Means inertia. For Hierarchical Clustering, the dendrogram is the primary tool for understanding the hierarchy and selecting a cluster cut. Clustering metrics can then be used to evaluate the result."

---

### Q16. What metrics do you use?

**Answer:**

> "I use Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Score."

```text
Silhouette          → Higher
Davies-Bouldin      → Lower
Calinski-Harabasz   → Higher
```

---

### Q17. Can we use `fit_predict()`?

**Answer:**

> "Yes. `fit_predict()` fits the Agglomerative Clustering model and returns the cluster labels for the data."

---

### Q18. Can we use `model.predict()` for new data?

**Answer:**

> "Standard sklearn AgglomerativeClustering does not provide the same direct `predict()` method for unseen data that K-Means provides."

---

### Q19. Why doesn't it work like K-Means for new data?

**Answer:**

> "K-Means learns explicit centroids that can be used to assign a new point. Hierarchical clustering builds a hierarchy of relationships and merges, rather than learning a set of reusable centroids."

---

### Q20. Why would you choose Hierarchical instead of K-Means?

**Answer:**

> "I would choose Hierarchical Clustering when understanding hierarchical relationships and the structure of groups is important, especially when a dendrogram is useful for exploring different clustering levels."

---

### Q21. When would you prefer K-Means?

**Answer:**

> "If I need efficient clustering and frequent assignment of new observations to existing clusters, K-Means is generally more convenient because it learns centroids and supports direct prediction."

---

### Q22. What is chaining?

**Answer:**

> "Chaining is a common issue with single linkage where clusters can become connected through a sequence of nearby observations, resulting in elongated clusters."

---

# 33. Perfect Interview Explanation

If an interviewer says:

> **"Explain how you implemented Hierarchical Clustering."**

You can say:

> **"First, I identified the problem as unsupervised clustering because the dataset didn't have a target variable and the objective was to discover natural groups. I selected the relevant numerical features and scaled them because hierarchical clustering is distance-based. I then used Agglomerative Hierarchical Clustering with an appropriate linkage method, such as Ward. I generated a dendrogram to understand the hierarchical structure and selected a suitable cluster cut. I then used the selected number of clusters with `AgglomerativeClustering` and obtained the cluster labels using `fit_predict()`. Finally, I evaluated the clustering using Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Score and interpreted the resulting groups. If frequent prediction of new observations is required, I would consider K-Means because standard sklearn AgglomerativeClustering doesn't provide the same direct `predict()` workflow."**

---

# 34. Final Memory

```text
NO TARGET
   ↓
UNSUPERVISED
   ↓
NEED GROUPS
   ↓
CLUSTERING
   ↓
HIERARCHICAL
   ↓
SCALE
   ↓
LINKAGE
   ↓
MERGE CLOSEST CLUSTERS
   ↓
DENDROGRAM
   ↓
HORIZONTAL CUT
   ↓
NUMBER OF CLUSTERS
   ↓
FIT_PREDICT()
   ↓
CLUSTER LABELS
   ↓
METRICS
   ↓
INTERPRET CLUSTERS
```

### One sentence to remember

> **"K-Means learns centroids and repeatedly assigns points to the nearest centroid, whereas Hierarchical Clustering builds a hierarchy by repeatedly merging clusters based on a linkage rule and uses a dendrogram to understand and select the final clustering."**

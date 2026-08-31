#DBSCAN = Density-Based Spatial Clustering of Applications with Noise

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("../data/customer_clustering.csv")


# ============================================================
# 2. SELECT FEATURES
# ============================================================

x = df[
    [
        "Age",
        "Annual_Income",
        "Spending_Score"
    ]
]


# ============================================================
# 3. SCALE DATA
# ============================================================

scaler = StandardScaler()

x_scaled = scaler.fit_transform(x)


# ============================================================
# 4. FIND eps USING K-DISTANCE
# ============================================================

min_samples = 6

neighbors = NearestNeighbors(
    n_neighbors=min_samples
)

neighbors.fit(x_scaled)

distances, indices = neighbors.kneighbors(x_scaled)

k_distances = sorted(
    distances[:, -1]
)


plt.figure(figsize=(10, 7))

plt.plot(k_distances)

plt.title("K-Distance Plot")

plt.xlabel("Customers")

plt.ylabel("Distance")

plt.show()


# ============================================================
# 5. CREATE DBSCAN MODEL
# ============================================================

# Choose eps from the elbow of the K-Distance plot

model = DBSCAN(
    eps=0.38,
    min_samples=min_samples
)


# ============================================================
# 6. FIT MODEL AND GET CLUSTERS
# ============================================================

labels = model.fit_predict(x_scaled)


# Add cluster labels to dataframe

df["Cluster"] = labels

print(df.head())


# ============================================================
# 7. CHECK NUMBER OF CLUSTERS
# ============================================================

number_of_clusters = len(
    set(labels)
) - (1 if -1 in labels else 0)

print(
    "Number of Clusters : ",
    number_of_clusters
)


# ============================================================
# 8. CHECK NOISE
# ============================================================

number_of_noise = (labels == -1).sum()

print(
    "Number of Noise Points : ",
    number_of_noise
)


# ============================================================
# 9. METRICS
# ============================================================

# Remove noise points before calculating metrics

mask = labels != -1

x_clustered = x_scaled[mask]

labels_clustered = labels[mask]


if len(set(labels_clustered)) >= 2:

    score_sil = silhouette_score(
        x_clustered,
        labels_clustered
    )

    print(
        "Silhouette Score : ",
        score_sil
    )


    score_davies = davies_bouldin_score(
        x_clustered,
        labels_clustered
    )

    print(
        "Davies-Bouldin Score : ",
        score_davies
    )


    score_calinski = calinski_harabasz_score(
        x_clustered,
        labels_clustered
    )

    print(
        "Calinski-Harabasz Score : ",
        score_calinski
    )

else:

    print(
        "Metrics cannot be calculated because "
        "DBSCAN did not produce at least 2 clusters."
    )
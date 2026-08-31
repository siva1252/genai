import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt


# Load dataset
df = pd.read_csv("../data/customer_clustering.csv")


# Select features
X = df[['Age', 'Annual_Income', 'Spending_Score']]


# Standardize features
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# Create PCA model
pca = PCA(n_components=2)


# Fit PCA and transform the data
X_pca = pca.fit_transform(X_scaled)


# Create PCA dataframe
df_pca = pd.DataFrame(
    X_pca,
    columns=['PC1', 'PC2']
)


# Explained variance
print("________ Explained Variance ________")

print("PC1:", pca.explained_variance_ratio_[0])
print("PC2:", pca.explained_variance_ratio_[1])

print(
    "Total:",
    pca.explained_variance_ratio_.sum()
)


# Show transformed data
print("________ PCA Data ________")

print(df_pca.head())


# Visualization
plt.figure(figsize=(8, 6))

plt.scatter(
    df_pca['PC1'],
    df_pca['PC2']
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA - Customer Dataset")

plt.show()
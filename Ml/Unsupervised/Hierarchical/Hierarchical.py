import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score,davies_bouldin_score,calinski_harabasz_score



df = pd.read_csv("../data/customer_clustering.csv")


x=df[['Age','Annual_Income','Spending_Score']]


scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)



#we are finding the distance between the customers like [a+b,a+c,b+c]

z=linkage(x_scaled,method='ward')

#ward is the method to find the distance between the customers

# plt.figure(figsize=(10, 7))
# dendrogram(z)
# plt.title("Dendrogram")
# plt.xlabel("Customers")
# plt.ylabel("Distance")
# plt.show()

model = AgglomerativeClustering(n_clusters=3,metric='euclidean',linkage='ward')
labels = model.fit_predict(x_scaled)

# df["Cluster"] = labels
# print(df.head())



print("_______________________________________metrics-----------------------------____")

score_sil = silhouette_score(x_scaled, labels)
print("Silhouette Score : ",score_sil)
score_davies = davies_bouldin_score(x_scaled, labels)
print("Davies-Bouldin Score : ",score_davies)
score_calinski = calinski_harabasz_score(x_scaled, labels)
print("Calinski-Harabasz Score : ",score_calinski)






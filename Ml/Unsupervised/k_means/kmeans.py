import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score,davies_bouldin_score,calinski_harabasz_score
import matplotlib.pyplot as plt



df = pd.read_csv("../data/customer_clustering.csv")

#print(df.head())
# print(df.shape)

x=df[['Age','Annual_Income','Spending_Score']]

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

inertias = []
silhouette_scores = []

for k in range(2, 11):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(x_scaled)

    inertias.append(model.inertia_)
    silhouette_scores.append(silhouette_score(x_scaled, model.labels_))

# results = pd.DataFrame({
#     "K": range(2, 11),
#     "Inertia": inertias,
#     "Silhouette": silhouette_scores     for best k from silhouette
# })

# print(results)



# final model (example: choose best K from silhouette)
model = KMeans(n_clusters=3, random_state=42, n_init=10)
model.fit(x_scaled)
# labels = model.labels_
# df["Cluster"] = labels
labels = model.fit_predict(x_scaled)
#silhouette_score = silhouette_score(x_scaled, labels)
# print("Silhouette Score : ",silhouette_score)


#new data

new_customer = [[50, 90000, 80]]

new_customer_scaled = scaler.transform(
    new_customer
)

prediction = model.predict(
    new_customer_scaled
)

print("Cluster:", prediction[0])



#new data prediction using csv file


# new_df = pd.read_csv("new_customers.csv")

# new_X = new_df[[
#     "Age",
#     "Annual_Income",
#     "Spending_Score"
# ]]

# new_X_scaled = scaler.transform(new_X)

# new_df["Cluster"] = model.predict(new_X_scaled)



print("------------metrics------------")

score_sil = silhouette_score(x_scaled, labels)
print("Silhouette Score : ",score_sil)

score_davies = davies_bouldin_score(x_scaled, labels)
print("Davies-Bouldin Score : ",score_davies)

score_calinski = calinski_harabasz_score(x_scaled, labels)
print("Calinski-Harabasz Score : ",score_calinski)


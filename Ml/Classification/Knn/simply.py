import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,roc_auc_score


df = pd.read_csv("../../data_classification/simple_classification_students.csv")


x = df[['Study_Hours']]
y = df['Result']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


model = KNeighborsClassifier(n_neighbors=3 , metric="euclidean" , weights="uniform")

model.fit(x_train_scaled, y_train)

y_pred = model.predict(x_test_scaled)

print("predictions : ",y_pred)
print("actual : ",y_test.to_numpy())
print("accuracy : ",accuracy_score(y_test, y_pred))


print("------------------metrics------------------")


confusion_matrix(y_test, y_pred)
print("confusion matrix : ",confusion_matrix(y_test, y_pred))

accuracy_score(y_test,y_pred)
print("accuracy : ",accuracy_score(y_test,y_pred))

precision_score(y_test,y_pred)
print("precision : ",precision_score(y_test,y_pred))

recall_score(y_test,y_pred)
print("recall : ",recall_score(y_test,y_pred))

f1_score(y_test,y_pred)
print("f1 score : ",f1_score(y_test,y_pred))

#roc-auc score ;
roc_auc=roc_auc_score(y_test,y_pred)

print("roc-auc score : ",roc_auc)

#probabaility score;

y_proba = model.predict_proba(x_test_scaled)[:,1]
print("probability : ",y_proba)
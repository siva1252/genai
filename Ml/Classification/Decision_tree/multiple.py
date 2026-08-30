import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,roc_auc_score



df=pd.read_csv("../../data_classification/multiple_classification_students.csv")

x=df[['Study_Hours','Previous_Marks','Attendance_Percent']]
y=df['Result']


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=DecisionTreeClassifier(criterion='gini',random_state=42)

model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("predicted class: ",y_pred)
print("actual class: ",y_test.to_numpy())



print("________________metrics________________")

confusion_metrics=confusion_matrix(y_test,y_pred)
print("confusion matrix: ",confusion_metrics)

accuracy=accuracy_score(y_test,y_pred)
print("accuracy: ",accuracy)

precision=precision_score(y_test,y_pred)
print("precision: ",precision)

recall=recall_score(y_test,y_pred)
print("recall: ",recall)

f1=f1_score(y_test,y_pred)
print("f1: ",f1)

#if we need to ROC-AUC score we can change your model.predict() to model.predict_proba() beacyse this things need probability not class

y_proba=model.predict_proba(x_test)[:,1]
print("probability: ",y_proba)

roc_auc=roc_auc_score(y_test,y_proba)
print("roc_auc: ",roc_auc)


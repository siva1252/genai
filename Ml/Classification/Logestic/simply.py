import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,roc_auc_score


df = pd.read_csv("../../data_classification/simple_classification_students.csv")

#print(df.head())

x = df[['Study_Hours']]
y = df['Result']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LogisticRegression()

model.fit(x_train, y_train)


y_pred = model.predict(x_test)

print("predicted class: ",y_pred)
print("actual class: ",y_test.to_numpy())



print("---------------------------metrics---------------------------")
confusion_metrics = confusion_matrix(y_test,y_pred)
print("confusion matrix: ",confusion_metrics)

accuracy = accuracy_score(y_test,y_pred)
print("accuracy: ",accuracy)

precision = precision_score(y_test,y_pred)
print("precision: ",precision)

recall = recall_score(y_test,y_pred)
print("recall: ",recall)

f1 = f1_score(y_test,y_pred)
print("f1: ",f1)

#if we need to ROC-AUC score we can change your model.predict() to model.predict_proba() beacyse this things need probability not class

y_proba = model.predict_proba(x_test)[:,1]
print("probability: ",y_proba)


roc_auc = roc_auc_score(y_test,y_proba)
print("roc_auc: ",roc_auc)


print("Total rows:", len(df))
print("Training rows:", len(x_train))
print("Testing rows:", len(x_test))

#inside model.fit() we have the following parameters:
#X_train + y_train
 #      ↓
#Logistic Regression
     #  ↓
#learns w and b
    #   ↓
#calculates z = wx + b
       # ↓
#sigmoid → probability
       # ↓
#Log Loss
       # ↓
#optimization adjusts w,b
       # ↓
#repeat
       # ↓
#final/best w,b

# print(model.coef_)
# print(model.intercept_)
# print(model.predict(x_test))


#-------------------------------------------------------------------------

#fro your obaservation to check formaules for this code don't think about this thing

#this tihngs are working in inside model.fit()
# w = model.coef_[0][0]
# b = model.intercept_[0]

# z = w * x_test["Study_Hours"].values + b

# # print("w:", w)
# # print("b:", b)
# # print("z:", z)

# import numpy as np

# probability = 1 / (1 + np.exp(-z))

# # print("z:", z)
# # print("probability:", probability)

# threshold = 0.5

# y_pred_manual = (probability >= threshold).astype(int)

# # print("threshold:", threshold)
# # print("predicted class:", y_pred_manual)

# print("Actual class:", y_test.to_numpy())
# print("Predicted class:", y_pred_manual)

#-------------------------------------------------------------------------
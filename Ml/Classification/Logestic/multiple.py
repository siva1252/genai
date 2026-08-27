import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,roc_auc_score

df=pd.read_csv("../../data_classification/multiple_classification_students.csv")

x=df[['Study_Hours','Previous_Marks','Attendance_Percent']]
y=df['Result']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)

model=LogisticRegression()


model.fit(x_train_scaled,y_train)
y_pred=model.predict(x_test_scaled)

print("predicted_class: ",y_pred)
print("actual class: ",y_test.to_numpy())
print("accuracy: ",model.score(x_test_scaled,y_test))

#With one feature:
#z=w1​x1​+b

#With multiple features:
#z=w1​x1​+w2​x2​+w3​x3​+b

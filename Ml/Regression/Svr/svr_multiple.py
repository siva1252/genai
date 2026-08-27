import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,root_mean_squared_error,r2_score,mean_absolute_error


df=pd.read_csv('../data/multiple_linear_regression_students.csv')

x=df[['Study_Hours','Previous_Marks','Attendance_Percent']]
y=df['Final_Marks']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)


model = SVR(kernel='linear',C=1.0,epsilon=0.1)

model.fit(x_train_scaled,y_train)
y_pred=model.predict(x_test_scaled)
# print("predicted marks: ",y_pred)
# print("--------------------------------")
# print("actual marks: ",y_test)
# print("--------------------------------")
# print("accuracy: ",model.score(x_test_scaled,y_test))

print("________metrics________")

mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
rmse=root_mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("mean absolute error: ",mae)
print("mean squared error: ",mse)
print("root mean squared error: ",rmse)
print("r2 score: ",r2)



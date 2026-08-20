import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error,root_mean_squared_error,r2_score,mean_absolute_error

df = pd.read_csv('../data/simple_linear_regression_students.csv')

x=df[['Study_Hours']]
y=df['Marks']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


model=KNeighborsRegressor(n_neighbors=2,metric="euclidean")


model.fit(x_train,y_train)


y_pred=model.predict(x_test)
print("predicted marks: ",y_pred)
print("--------------------------------")
print("actual marks: ",y_test)
# 


print("__________________________Metrics__________________________")
mse=mean_squared_error(y_test,y_pred)
rmse=root_mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
print("mean squared error: ",mse)
print("root mean squared error: ",rmse)
print("r2 score: ",r2)
print("mean absolute error: ",mae)
print("--------------------------------")
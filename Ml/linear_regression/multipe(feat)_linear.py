import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,root_mean_squared_error,r2_score,mean_absolute_error

df = pd.read_csv('data/multiple_linear_regression_students.csv')

x=df[['Study_Hours','Previous_Marks','Attendance_Percent']]
y=df['Final_Marks']

x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(x_train,y_train)

# print("Intercept: ",model.intercept_)
# print("Coefficient: ",model.coef_)


y_pred = model.predict(x_test)




# print("predicted marks",y_pred)
# print("actual marks",y_test)

mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
rmse = root_mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

print("Mean Absolute Error: ",mae)
print("Mean Squared Error: ",mse)
print("Root Mean Squared Error: ",rmse)
print("R2 Score: ",r2)
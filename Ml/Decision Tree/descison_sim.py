import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_text
from sklearn.metrics import mean_squared_error,root_mean_squared_error,r2_score,mean_absolute_error




df = pd.read_csv('../data/simple_linear_regression_students.csv')


X = df[['Study_Hours']]
y = df['Marks']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = DecisionTreeRegressor()

model.fit(X_train,y_train)

# print("model score: ",model.score(X_test,y_test))

y_pred = model.predict(X_test)
# print("predicted marks: ",y_pred)
# print("actual marks: ",y_test)

#test checking
# tree_rules = export_text(
#     model,
#     feature_names=["Study_Hours"]
# )

# print(tree_rules)


mse = mean_squared_error(y_test,y_pred)
rmse = root_mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
mae = mean_absolute_error(y_test,y_pred)
print("mse: ",mse)
print("rmse: ",rmse)
print("r2: ",r2)
print("mae: ",mae)
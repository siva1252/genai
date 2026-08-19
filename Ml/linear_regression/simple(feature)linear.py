import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,root_mean_squared_error,r2_score,mean_absolute_error


df = pd.read_csv('../data/simple_linear_regression_students.csv')

# print(df)
# print(df.head())
# print(df.shape)
# print(df.info())
# print(df.columns)


x=df[['Study_Hours']]
y=df['Marks']


x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.2,random_state=42)

# print("x train: ",x_train)
# print("x test: ",x_test)










# print("y train: ",y_train)
# print("y test: ",y_test)


model = LinearRegression()

model.fit(x_train,y_train)

# print("Intercept: ",model.intercept_)
# print("Coefficient: ",model.coef_[0])


y_pred = model.predict(x_test)
print("predicted marks",y_pred)

mse = mean_squared_error(y_test,y_pred)
rmse = root_mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
mae = mean_absolute_error(y_test,y_pred)
print("Mean Squared Error: ",mse)
print("Root Mean Squared Error: ",rmse)
print("R2 Score: ",r2)
print("Mean Absolute Error: ",mae)
































































#formula
 #we find this intercept and coefficient by using the formula
      #model.fit 
   

#prediction label=intercept+coefficient*feature
  #Error = Actual Label - Predicted Label



#we are starting metrices
#Mean Squared Error (MSE) it tell us the average size of our prediction errors.
#Why use it?
#To understand how accurate our predictions are in the original unit.
#MSE = (1/n) × Σ (yᵢ - ŷᵢ)²

#Actual - Predicted
       #↓
#Square
      # ↓
#Average
     


#Root Mean Squared Error (RMSE) it tell us the average size of our prediction errors.
#Why use it?
#To understand how accurate our predictions are in the original unit.
#RMSE = √MSE

#MSE
      #↓
#Square root
      #↓
#Back to original unit



#r2  How much target variation the model explains
#R² = 1 - [Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)²]
# Model's squared error
#         ↓
# versus
# Error from simply predicting the average


#mean absolute error (MAE) it tell us the average size of our prediction errors.
#MAE = (1/n) × Σ |yᵢ - ŷᵢ|
#Actual - Predicted
       #↓
#Absolute value
       #↓
#Average

"""MAE
→ |error|
→ Average

MSE
→ error²
→ Average

RMSE
→ √MSE

R²
→ Model error compared with average-baseline error"""




#INISIDE MODEL.FIT()
#find coefficient
#→ predict
#→ error
#→ change coefficient
#→ predict again   


#Find b₀, b₁, b₂, b₃
        #↓
#that minimize
        #↓
#Σ(Actual - Predicted)²


#β = (XᵀX)⁻¹Xᵀy   Ordinary Least Squares (OLS).  Σ(Actual - Predicted)²
#β is the vector of coefficients
#X is the matrix of features
#y is the vector of labels
#this is multi-variable linear regression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error,root_mean_squared_error,r2_score,mean_absolute_error

df = pd.read_csv('../data/multiple_linear_regression_students.csv')

x=df[['Study_Hours','Previous_Marks','Attendance_Percent']]
y=df['Final_Marks']


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model = DecisionTreeRegressor()

model.fit(x_train,y_train)

y_pred=model.predict(x_test)
print("predicted marks: ",y_pred)
print("actual marks: ",y_test)

mse = mean_squared_error(y_test,y_pred)
rmse = root_mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
mae = mean_absolute_error(y_test,y_pred)
print("mse: ",mse)
print("rmse: ",rmse)
print("r2: ",r2)
print("mae: ",mae)






"""in descion tree first we check avergae of label and after that we can check impurity like what we get average - label whole square and again avergae so we get imputiy"""
  #DecisionTreeRegressor(criterion="squared_error")
#this is the formula for impurity
#I(node) = (1/n) Σ(yᵢ - ȳ)²
#after that we check featrues this thing is It considers boundaries between neighboring sorted values:
#like between of that value which we can get from the data possible thresholds
#It finds the split with the lowest resulting weighted impurity.

#so we find perfect threshold for the features

#it should be check know again tree structure to choose the split with the largest impurity reduction.

#after thgata left and right side of tress is agian check and again based on that process 
#so left and right side also it will be do again avergae it finds so based on that avergar also it find again impurity 
##I(node) = (1/n) Σ(yᵢ - ȳ)² of this formaule 

#agian after that calcaute that split's impurity
'''I(split)
=
(n_left/n) × I(left)
+
(n_right/n) × I(right)'''

#so after this spliting before after impurity one and know this one also again we check gain
'''Gain
=
Parent impurity - Split impurity'''

'''
Decision Tree:
Smaller resulting impurity → better split
                       ↓
              Larger Gain '''



'''Parent impurity → impurity before splitting.
Child impurity / weighted child impurity → impurity after a particular split.
Split impurity → you can use this term to refer to the resulting weighted impurity, but "weighted child impurity" is clearer.
Gain / impurity reduction → difference between parent impurity and weighted child impurity.
Parent Impurity
       ↓
Try a split
       ↓
Weighted Child Impurity
       ↓
Gain = Parent - Child
       ↓
Compare all splits
       ↓
Smaller Child Impurity
= Larger Gain
= Better Split'''
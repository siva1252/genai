# KNN Regression — Interview Q&A

Interview-ready notes covering **basic concept → calculation → code → scaling → hyperparameters → evaluation → interview answers**.

---

## Q1. What is KNN Regression?

**KNN (K-Nearest Neighbors) Regression is a supervised learning algorithm that predicts a continuous/numeric target by finding the `K` nearest training samples and averaging their target values.**

Example:

```text
New student
Study_Hours = 5
        ↓
Find nearest training students
        ↓
K = 2
        ↓
Student 1 → Marks = 50
Student 2 → Marks = 60
        ↓
Average = (50 + 60) / 2
        ↓
Prediction = 55
```

---

## Q2. What does KNN mean?

**K = Number of nearest neighbors**  
**NN = Nearest Neighbors**

So:

> **KNN = K-Nearest Neighbors**

Example:

```text
K = 3
```

means:

> Use the **3 nearest training samples** to make the prediction.

---

## Q3. Is KNN Regression classification or regression?

KNN can be used for both.

**KNN Regression**

Target is numeric/continuous:

```text
40
55
72.5
90
```

We normally **average** the target values of the nearest neighbors.

**KNN Classification**

Target represents classes:

```text
Cat
Dog
Cat
```

We select the most common class among the nearest neighbors.

For our current learning, we use:

```python
KNeighborsRegressor
```

so we are doing **regression**.

---

## Q4. What happens before using KNN?

The normal ML workflow is:

```text
Dataset
   ↓
Read data
   ↓
X = Features
y = Target
   ↓
Train/Test Split
   ↓
Scale features if required
   ↓
Create KNN model
   ↓
model.fit()
   ↓
model.predict()
   ↓
Evaluation metrics
```

---

## Q5. What is `X` and `y`?

```python
X = df[
    [
        "Study_Hours",
        "Previous_Marks",
        "Attendance_Percent"
    ]
]

y = df["Final_Marks"]
```

So:

```text
X → input features
y → target/output
```

Example:

```text
Study_Hours       → Feature
Previous_Marks    → Feature
Attendance        → Feature

Final_Marks       → Target
```

---

## Q6. Why do we split the data?

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

We split because:

```text
Training data
→ used to train the model

Testing data
→ used to check the model on unseen data
```

Remember:

> **Training → Build/prepare the model**  
> **Testing → Check the model**

---

## Q7. What happens in `model.fit()` for KNN?

```python
model.fit(X_train, y_train)
```

KNN is different from Decision Tree and Random Forest.

It does **not build a tree**.

It mainly stores the training examples:

```text
X_train + y_train
       ↓
     KNN
       ↓
Stores training information
```

For example:

```text
Study_Hours → Marks

2 → 40
4 → 50
6 → 60
8 → 80
```

There is **no test data** involved in `fit()`.

---

## Q8. Where does the new/test data enter?

Here:

```python
y_pred = model.predict(X_test)
```

This is where KNN starts finding neighbors.

```text
X_test
   ↓
Compare with X_train
   ↓
Calculate distances
   ↓
Find nearest neighbors
   ↓
Use their y_train values
   ↓
Prediction
```

---

## Q9. How does KNN calculate distance with one feature?

Suppose:

```text
New data = 5
```

Training data:

```text
Study_Hours

2
4
6
8
10
```

For one feature:

```text
Distance = |x − y|
```

Calculations:

```text
|5 − 2|  = 3
|5 − 4|  = 1
|5 − 6|  = 1
|5 − 8|  = 3
|5 − 10| = 5
```

So:

```text
Training value    Distance

2                 3
4                 1
6                 1
8                 3
10                5
```

---

## Q10. How does KNN calculate distance with multiple features?

Suppose we have:

```text
Study_Hours
Previous_Marks
Attendance
```

Euclidean distance is:

```text
Distance = √[(x₁ − y₁)² + (x₂ − y₂)² + (x₃ − y₃)²]
```

Where:

```text
x = new data
y = one training row
```

Example:

```text
New:
Study_Hours = 5
Previous_Marks = 70
Attendance = 80

Training row:
Study_Hours = 4
Previous_Marks = 65
Attendance = 75
```

Then:

```text
Distance = √[(5 − 4)² + (70 − 65)² + (80 − 75)²]
         = √[1 + 25 + 25]
         = √51
         ≈ 7.14
```

KNN performs this calculation against **every training row**.

---

## Q11. What does `n_neighbors` mean?

```python
model = KNeighborsRegressor(
    n_neighbors=2
)
```

`n_neighbors` means:

> **How many nearest training samples should be used for prediction?**

Example:

```text
K = 2
```

means choose the **2 smallest distances**.

---

## Q12. How does KNN make a regression prediction?

Suppose:

```text
Distance    Target

0.5         50
0.8         60
2.0         90
4.0         100
```

If:

```text
K = 2
```

select:

```text
0.5 → 50
0.8 → 60
```

Then:

```text
Prediction = (50 + 60) / 2 = 55
```

So:

> KNN uses distance only to find the nearest rows. It then uses the target values of those rows to calculate the prediction.

---

## Q13. Does KNN average the distances?

**No.**

This is important.

It does:

```text
Distance
   ↓
Find smallest K
   ↓
Identify those rows
   ↓
Take their TARGET values
   ↓
Average target values
```

Example:

```text
Distance → 1 → Target = 50
Distance → 2 → Target = 60
```

We average:

```text
50 and 60
```

**not:**

```text
1 and 2
```

---

## Q14. What is Feature Scaling?

> Feature scaling transforms feature values into comparable numerical ranges so that a large-scale feature does not dominate the distance calculation.

Example:

```text
Study_Hours → 1–10
Marks       → 0–100
Salary      → 20,000–100,000
```

Without scaling, a large numerical feature can dominate the distance.

---

## Q15. Why is scaling important for KNN?

Because KNN is **distance-based**.

Example:

```text
Study_Hours difference = 2
Salary difference      = 20,000
```

The salary difference can dominate the distance.

After scaling:

```text
Study_Hours → comparable range
Salary      → comparable range
```

Now the features can contribute more fairly.

---

## Q16. Is scaling required for a single feature?

Usually it is **not necessary** for KNN when there is only one feature.

Why?

Because there isn't another feature with a different scale competing with it.

With multiple features:

```text
Feature 1 → 1–10
Feature 2 → 0–100
Feature 3 → 20,000–100,000
```

scaling becomes important.

---

## Q17. How does `StandardScaler` work?

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

`StandardScaler` transforms the features based on their mean and standard deviation.

Conceptually:

```text
z = (x − μ) / σ
```

Where:

```text
x = original value
μ = training mean
σ = training standard deviation
```

---

## Q18. Why do we use `fit_transform()` on training data?

```python
X_train_scaled = scaler.fit_transform(X_train)
```

Because we want the scaler to:

```text
1. Learn scaling information from X_train
2. Transform X_train
```

---

## Q19. Why do we only use `transform()` on test data?

```python
X_test_scaled = scaler.transform(X_test)
```

Because test data must use the **same scaling information learned from training data**.

Correct:

```text
X_train
 ↓
fit_transform()

X_test
 ↓
transform()
```

Not:

```text
X_test
 ↓
fit_transform() ❌
```

---

## Q20. What is `metric="euclidean"`?

```python
model = KNeighborsRegressor(
    n_neighbors=2,
    metric="euclidean"
)
```

It tells KNN:

> **Use Euclidean distance to determine how far one data point is from another.**

For multiple features:

```text
Distance = √[(x₁ − y₁)² + (x₂ − y₂)² + ⋯ ]
```

---

## Q21. Is `metric="euclidean"` mandatory?

**No.**

This:

```python
model = KNeighborsRegressor(
    n_neighbors=2
)
```

already uses Euclidean distance by default.

Writing:

```python
metric="euclidean"
```

just explicitly tells the model which distance metric to use.

---

## Q22. Is KNN `metric` the same as MAE/MSE/RMSE/R²?

**No.**

They have completely different purposes.

**KNN distance metric**

```text
metric="euclidean"
        ↓
Used internally
        ↓
Find nearest neighbors
```

**Evaluation metrics**

```text
y_pred + y_test
       ↓
MAE
MSE
RMSE
R²
       ↓
Evaluate model performance
```

Remember:

> **Euclidean → find neighbors**  
> **MAE/MSE/RMSE/R² → evaluate predictions**

---

## Q23. What is `weights`?

`weights` decides **how much influence each selected neighbor has** on the prediction.

Two common options:

```python
weights="uniform"
```

and:

```python
weights="distance"
```

---

## Q24. What is `weights="uniform"`?

Every selected neighbor gets equal importance.

Example:

```text
Neighbor 1 → 50
Neighbor 2 → 60
Neighbor 3 → 90
```

Prediction:

```text
(50 + 60 + 90) / 3 = 66.67
```

---

## Q25. What is `weights="distance"`?

Closer neighbors receive more influence.

Conceptually:

```text
Weight = 1 / Distance
```

Example:

```text
Distance    Target

1           50
2           60
5           90
```

The first neighbor gets more influence because it is closer.

---

## Q26. Do we definitely need `weights`?

**No.**

This is perfectly valid:

```python
model = KNeighborsRegressor(
    n_neighbors=5
)
```

By default, KNN uses uniform weighting.

Use:

```python
weights="distance"
```

only when testing shows that distance-based weighting gives better validation performance.

---

## Q27. What happens when `K` is very small?

Small `K` means very few neighbors influence the prediction.

Example:

```text
K = 1
```

Only one training sample determines the prediction.

This makes the model very sensitive to individual training samples.

➡️ **Can cause overfitting.**

---

## Q28. What happens when `K` is very large?

Large `K` means many neighbors influence the prediction.

Example:

```text
K = 50
```

The prediction becomes very averaged/smooth.

➡️ **Can cause underfitting.**

---

## Q29. How do we identify overfitting in KNN?

We compare training and validation/test performance.

Example:

```text
K = 1

Training RMSE = 0.5
Test RMSE     = 12
```

```text
Training → very good
Test      → poor
```

➡️ **Possible overfitting**

---

## Q30. How do we identify underfitting in KNN?

Example:

```text
K = 50

Training RMSE = 14
Test RMSE     = 15
```

Both errors are high:

```text
Training → poor
Test      → poor
```

➡️ **Possible underfitting**

---

## Q31. How do we find a good K?

Test multiple K values.

Example:

```text
K = 1
K = 3
K = 5
K = 7
K = 10
```

Then compare validation performance.

| K  | Train RMSE | Validation RMSE |
| -: | ---------: | --------------: |
|  1 |        0.5 |              12 |
|  3 |          2 |               7 |
|  5 |          3 |               5 |
| 10 |          5 |               6 |
| 50 |         14 |              15 |

Here `K=5` gives a good balance.

So:

> We don't randomly decide K. We test different values and choose a value that generalizes well.

---

## Q32. What happens in `model.predict()`?

```python
y_pred = model.predict(X_test)
```

For every test row:

```text
X_test
   ↓
Compare with every X_train row
   ↓
Calculate distance
   ↓
Find K nearest rows
   ↓
Take their y_train values
   ↓
Average / weighted average
   ↓
y_pred
```

It **does not train again**.

---

## Q33. Does test data participate in training?

**No.**

```text
X_train + y_train
       ↓
     fit()
       ↓
Training information
```

Then:

```text
X_test
   ↓
predict()
   ↓
y_pred
```

Then:

```text
y_pred vs y_test
       ↓
Evaluation
```

Remember:

> **TRAIN → BUILD/PREPARE**  
> **TEST → CHECK**

---

## Q34. How do we evaluate KNN Regression?

We use:

```text
MAE
MSE
RMSE
R²
```

General interpretation:

```text
MAE  ↓ better
MSE  ↓ better
RMSE ↓ better
R²   ↑ better
```

Example:

```python
mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)
```

---

## Q35. Complete KNN code — Multiple Features

```python
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score
)

df = pd.read_csv(
    "../data/multiple_linear_regression_students.csv"
)

X = df[
    [
        "Study_Hours",
        "Previous_Marks",
        "Attendance_Percent"
    ]
]

y = df["Final_Marks"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = KNeighborsRegressor(
    n_neighbors=2
)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print("Predicted marks:", y_pred)
print("--------------------------------")
print("Actual marks:", y_test.values)

mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R²:", r2)
```

---

## Q36. Explain the complete KNN workflow

This is the **most important flow to remember**:

```text
Dataset
   ↓
Read data
   ↓
X = Features
y = Target
   ↓
Train/Test Split
   ↓
X_train + y_train
X_test  + y_test
   ↓
Scale X_train
   ↓
Scale X_test using SAME scaler
   ↓
Create KNN
   ↓
n_neighbors = K
   ↓
model.fit(X_train, y_train)
   ↓
Training data stored
   ↓
model.predict(X_test)
   ↓
For each test row:
   ↓
Compare with every X_train row
   ↓
Calculate Euclidean distance
   ↓
Find K smallest distances
   ↓
Take corresponding y_train values
   ↓
Average / weighted average
   ↓
y_pred
   ↓
Compare y_pred with y_test
   ↓
MAE / MSE / RMSE / R²
```

---

## Q37. Interview: Explain KNN Regression

Strong interview answer:

> **KNN Regression is a supervised learning algorithm used to predict continuous values. For a new data point, it calculates the distance between the new point and the training samples, selects the K nearest samples, and uses their target values to calculate the prediction, normally by averaging them. Since KNN is distance-based, feature scaling is important when features have different ranges. During `fit()`, KNN stores the training data, while during `predict()`, it finds the nearest neighbors and generates the prediction.**

---

## Q38. One-line interview answers

**What is KNN?**  
A supervised algorithm that predicts using the nearest training samples.

**What does K mean?**  
The number of nearest neighbors used for prediction.

**What is `n_neighbors`?**  
It specifies K.

**Why is scaling important in KNN?**  
Because KNN uses distance, and large-scale features can dominate the distance calculation.

**What is Euclidean distance?**  
A distance formula used to measure how far two data points are from each other.

**What does `metric="euclidean"` do?**  
It tells KNN to use Euclidean distance when finding neighbors.

**What does `weights="uniform"` mean?**  
All selected neighbors have equal influence.

**What does `weights="distance"` mean?**  
Closer neighbors have more influence.

**What happens in `fit()`?**  
KNN stores the training data; it does not build a decision tree.

**What happens in `predict()`?**  
It calculates distances, finds K nearest neighbors, and generates the prediction.

**Does KNN train again during `predict()`?**  
No.

**What happens with very small K?**  
It can become sensitive to individual samples and overfit.

**What happens with very large K?**  
It can become too smooth and underfit.

**How do you choose K?**  
Test different K values and select one with good validation performance.

**How do you evaluate KNN Regression?**  
MAE, MSE, RMSE and R².

**What is the difference between KNN metric and evaluation metrics?**  
KNN's distance metric finds neighbors; MAE, MSE, RMSE and R² evaluate the final predictions.

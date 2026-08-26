Linear Regression
→ coefficients + intercept

Decision Tree
→ thresholds + impurity + gain

Random Forest
→ multiple Decision Trees + averaging

Knn model 
-- finding the `K` nearest training samples and averaging their target values

#in code we write this things in from sklearn
linear_model
→ Linear-based algorithms

tree
→ Tree-based algorithms

ensemble
→ Multiple models combined together



#regressions of descion and random 
max_depth controls how deep the tree can grow.
min_samples_split checks whether the current node has enough samples to split.
min_samples_leaf checks whether both children created by a split have enough samples. If not, another threshold can be considered.
max_features randomly selects how many features are considered at each split.



Linear Regression
→ "Find the best line by minimizing error."

Decision Tree
→ "Find the best split by reducing impurity."

Random Forest
→ "Build many trees and average them."

KNN
→ "Find nearby examples and average their targets."

SVR
→ "Find a regression function with an acceptable-error tube
   and penalize errors outside that tube."



   --------------------------------
   compare all metrics like this 
   5 Algorithms
     ↓
Same dataset
     ↓
Same training data
     ↓
Same validation/test data
     ↓
Get metrics for every model
     ↓
Compare same metric with same metric
     ↓
Find best MAE
Find best MSE
Find best RMSE
Find best R²
     ↓
Check generalization / overfitting
     ↓
Choose final model

----------------------------------------------------

# Regression Algorithms — Interview-Level Explanation

### 1. Linear Regression

> **Linear Regression is a supervised learning algorithm used to predict a continuous target by learning a linear relationship between the input features and the target using coefficients and an intercept.**

For one feature:

$$
\hat y = wx+b
$$

For multiple features:

$$
\hat y=w_1x_1+w_2x_2+\cdots+w_nx_n+b
$$

During training, it finds the coefficients and intercept that minimize the prediction error, commonly using squared error.

**Example:** Predict `Final_Marks` from `Study_Hours`, `Previous_Marks`, and `Attendance`.

---

### 2. Decision Tree Regression

> **Decision Tree Regression is a supervised learning algorithm that predicts a continuous target by recursively splitting the training data using feature conditions and predicting a numeric value at each leaf.**

During training:

```text
Feature + threshold
        ↓
Left / Right
        ↓
Parent impurity
        ↓
Child impurity
        ↓
Weighted child impurity
        ↓
Gain
        ↓
Best split
        ↓
Repeat recursively
```

For regression, the leaf normally predicts the **mean target value of the samples reaching that leaf**.

**Example:**

```text
Study_Hours <= 5.5
       ↓
   /       \
 LEFT     RIGHT
```

---

### 3. Random Forest Regression

> **Random Forest Regression is an ensemble supervised learning algorithm that builds multiple Decision Trees using randomized training samples and feature selection, then combines their predictions, typically by averaging them.**

Training:

```text
Training data
      ↓
Bootstrap samples
      ↓
Tree 1
Tree 2
Tree 3
...
Tree N
      ↓
Each tree learns its own splits
```

Prediction:

```text
Tree 1 → 80
Tree 2 → 84
Tree 3 → 82
      ↓
Average
      ↓
82
```

$$
\hat y=\frac{1}{N}\sum_{i=1}^{N}\hat y_i
$$

**Main purpose:** Usually more stable and better generalized than a single Decision Tree.

---

### 4. KNN Regression

> **KNN Regression is a supervised learning algorithm that predicts a continuous target by finding the K nearest training samples to a new data point and using their target values, usually by averaging them.**

During prediction:

```text
New data
   ↓
Calculate distance to every training row
   ↓
Find K smallest distances
   ↓
Take corresponding target values
   ↓
Average
   ↓
Prediction
```

For multiple features, Euclidean distance can be:

$$
d=\sqrt{(x_1-y_1)^2+(x_2-y_2)^2+\cdots+(x_n-y_n)^2}
$$

**Example:**

```text
K = 3

Nearest targets:
50
60
70

Prediction = (50+60+70)/3 = 60
```

**Main point:** KNN does not build a tree or learn a regression equation like Linear Regression; it relies on neighboring training samples.

---

### 5. SVR

> **Support Vector Regression (SVR) is a supervised learning algorithm that learns a regression function while allowing prediction errors within an epsilon (`ε`) margin and penalizing errors that go beyond that margin.**

For a linear kernel:

$$
f(x)=w^Tx+b
$$

During training:

```text
Training data
      ↓
Learn regression function
      ↓
Create ε tolerance around it
      ↓
Calculate training errors
      ↓
Error <= ε
→ no epsilon loss

Error > ε
→ penalize excess error
      ↓
C controls penalty strength
      ↓
Optimization
      ↓
Final SVR model
```

**Example:**

```text
Prediction = 60
ε = 5
```

Allowed region:

```text
55 ───── lower
60 ───── regression function
65 ───── upper
```

If actual = `63`:

```text
Error = 3
3 <= 5
→ no epsilon loss
```

If actual = `72`:

```text
Error = 12
12 > 5
→ extra error = 7
→ penalty
```

**Main point:** SVR uses **epsilon-insensitive loss and optimization** to learn the regression function.

---

# Interview Comparison

| Algorithm             | Core idea                                           |
| --------------------- | --------------------------------------------------- |
| **Linear Regression** | Coefficients + intercept                            |
| **Decision Tree**     | Thresholds + impurity + gain                        |
| **Random Forest**     | Multiple Decision Trees + averaging                 |
| **KNN**               | Distance + K nearest neighbors + averaging          |
| **SVR**               | Regression function + epsilon margin + optimization |

### One-line interview answers

**Linear Regression:**

> Learns a linear relationship using coefficients and intercept to predict a continuous target.

**Decision Tree:**

> Recursively splits data using feature thresholds and predicts the mean target at the leaves.

**Random Forest:**

> Combines predictions from multiple randomized Decision Trees to produce a more stable regression model.

**KNN:**

> Predicts a target using the target values of the K nearest training samples.

**SVR:**

> Learns a regression function while ignoring errors within epsilon and penalizing errors outside that margin.


---------------------------------------------

# Regression — Important Concepts Only

## 1. Linear Regression

**Training:**
Learns **coefficients and intercept** that minimize prediction error.

**Prediction:**
Uses the learned equation:

$$
\hat y=w_1x_1+\cdots+w_nx_n+b
$$

**Scaling:**
Usually **not required**.

**Hyperparameters:**
Basic `LinearRegression` has very few important tuning parameters.

**Overfitting / Underfitting:**
Can happen depending on the data and model complexity; check training vs validation/test performance.

**Metrics:**
MAE, MSE, RMSE, R².

---

## 2. Decision Tree Regression

**Training:**
Finds the best **feature + threshold** using impurity reduction/gain.

```text
Parent impurity
→ possible splits
→ child impurity
→ weighted child impurity
→ gain
→ best split
→ repeat
```

**Prediction:**
New data follows the learned conditions until it reaches a **leaf**.

**Scaling:**
❌ Not required.

**Hyperparameters:**

```text
max_depth
min_samples_split
min_samples_leaf
```

**Overfitting / Underfitting:**

```text
Too deep → overfitting
Too shallow → underfitting
```

**Metrics:**
MAE, MSE, RMSE, R².

---

## 3. Random Forest Regression

**Training:**
Builds **multiple Decision Trees** using bootstrap samples and feature randomness.

**Prediction:**

```text
Tree 1 → prediction
Tree 2 → prediction
Tree 3 → prediction
      ↓
Average
      ↓
Final prediction
```

**Scaling:**
❌ Not required.

**Hyperparameters:**

```text
n_estimators
max_depth
min_samples_split
min_samples_leaf
max_features
```

**Overfitting / Underfitting:**
Can still overfit, but multiple trees generally make the model more stable.

**Metrics:**
MAE, MSE, RMSE, R².

---

## 4. KNN Regression

**Training:**
Mainly **stores the training data**; it does not build a tree or learn coefficients like Linear Regression.

**Prediction:**

```text
New data
→ calculate distance to training rows
→ find K nearest
→ take their target values
→ average
→ prediction
```

**Scaling:**
✅ Important when features have different ranges because KNN uses distance.

**Hyperparameters:**

```text
n_neighbors
weights
metric
```

**Overfitting / Underfitting:**

```text
Small K → can overfit
Large K → can underfit
```

**Metrics:**
MAE, MSE, RMSE, R².

---

## 5. SVR

**Training:**
Uses **optimization** to learn a regression function while using an **epsilon (`ε`) tolerance**.

```text
Learn function
→ calculate training errors
→ compare with ε
→ inside ε → no epsilon-loss
→ outside ε → penalty
→ optimization
→ final model
```

**Prediction:**
Uses the **already learned SVR function** to predict new data.

**Scaling:**
✅ Usually important, especially with different feature scales and kernel-based SVR.

**Hyperparameters:**

```text
kernel
C
epsilon
gamma
```

**Overfitting / Underfitting:**

```text
Parameter settings too flexible → overfitting
Too restrictive/simple → underfitting
```

**Metrics:**
MAE, MSE, RMSE, R².

---

# Final Memory

```text
Linear Regression
→ coefficients + intercept

Decision Tree
→ threshold + impurity + gain

Random Forest
→ many trees + average

KNN
→ distance + K neighbors + average

SVR
→ regression function + epsilon + optimization
```

And for all five:

```text
Training
→ learn according to the algorithm

Prediction
→ use the learned model on new data

Overfitting/Underfitting
→ compare training vs validation/test performance

Metrics
→ MAE ↓
→ MSE ↓
→ RMSE ↓
→ R² ↑
```

# Random Forest Regression — Interview Q&A

Interview-ready notes covering **what → why → how → formulas → practical code → interview answer**.

---

## Q1. What is Random Forest Regression?

**Random Forest Regression is a supervised learning algorithm that combines multiple Decision Trees and averages their predictions to predict a continuous/numeric target.**

Example:

```text
Study_Hours
Previous_Marks
Attendance
      ↓
Final_Marks
```

Random Forest creates multiple Decision Trees:

```text
Training Data
      ↓
 ┌────┼────┐
 ↓    ↓    ↓
Tree1 Tree2 Tree3
 ↓    ↓    ↓
 80   84   82
 └────┼────┘
      ↓
Average
      ↓
82
```

Final Prediction = (80 + 84 + 82) / 3 = **82**

---

## Q2. Why do we use Random Forest?

We use Random Forest because a **single Decision Tree can easily become too specific to the training data**.

Random Forest combines many trees.

```text
Single Decision Tree
        ↓
Can learn training data too specifically
        ↓
Overfitting
```

Random Forest:

```text
Tree 1 → 80
Tree 2 → 84
Tree 3 → 82
Tree 4 → 81
...
        ↓
Average
        ↓
More stable prediction
```

So Random Forest generally gives a **more stable and generalized model** than one Decision Tree.

---

## Q3. How is Random Forest different from Decision Tree?

**Decision Tree**

```text
Training data
     ↓
One Tree
     ↓
Best splits
     ↓
Leaf
     ↓
Prediction
```

**Random Forest**

```text
Training data
     ↓
Bootstrap samples
     ↓
Multiple Decision Trees
     ↓
Each tree learns its own splits
     ↓
Predictions from all trees
     ↓
Average
     ↓
Final prediction
```

**Interview answer:**

> A Decision Tree uses one tree, while Random Forest combines multiple Decision Trees using randomized samples and feature selection, then combines their predictions.

---

## Q4. What happens inside `model.fit()`?

```python
model.fit(X_train, y_train)
```

For Random Forest:

```text
Training Data
      ↓
Bootstrap sampling
      ↓
Create sample for Tree 1
      ↓
Build Decision Tree 1
      ↓
Bootstrap sampling
      ↓
Create sample for Tree 2
      ↓
Build Decision Tree 2
      ↓
...
      ↓
Build all trees
      ↓
Random Forest trained
```

Each individual tree uses the **Decision Tree split process**:

```text
Target average
      ↓
Parent impurity
      ↓
Possible features + thresholds
      ↓
Left / Right
      ↓
Weighted child impurity
      ↓
Gain
      ↓
Best split
      ↓
Repeat
      ↓
Complete Tree
```

---

## Q5. What is Bootstrap Sampling?

**Bootstrap Sampling means creating a training sample by randomly selecting rows from the original training dataset with replacement.**

Original data:

```text
Rows:
1  2  3  4  5
```

Tree 1 might receive:

```text
1  3  3  5  2
```

Tree 2 might receive:

```text
2  4  1  4  5
```

Tree 3 might receive:

```text
5  2  2  3  1
```

Notice:

```text
3 appears twice
4 may not appear
```

because sampling is **with replacement**.

So each tree gets a slightly different training sample.

---

## Q6. Why are rows repeated in Bootstrap Sampling?

Because Random Forest samples **with replacement**.

This creates different datasets for different trees.

```text
Original:
1 2 3 4 5
```

Possible samples:

```text
Tree 1 → 1 2 2 4 5
Tree 2 → 3 3 1 5 4
Tree 3 → 2 5 5 1 3
```

Now the trees don't all learn from exactly the same rows.

That creates **diversity between the trees**.

---

## Q7. What is `n_estimators`?

> `n_estimators` = number of Decision Trees in the Random Forest.

```python
RandomForestRegressor(n_estimators=3)
```

means:

```text
Tree 1
Tree 2
Tree 3
```

If:

```python
n_estimators=100
```

then:

```text
100 Decision Trees
      ↓
100 predictions
      ↓
Average
      ↓
Final prediction
```

---

## Q8. How does `random_state` work?

`random_state` controls the randomness so that you can reproduce the same result.

```python
RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
```

```text
random_state=42
       ↓
same random process
       ↓
same result
```

Without a fixed `random_state`, the random sampling can change between runs.

---

## Q9. How are multiple Decision Trees created?

Suppose:

```python
n_estimators=3
```

Random Forest creates:

```text
Original Training Data
        ↓
 ┌──────┼──────┐
 ↓      ↓      ↓
Sample1 Sample2 Sample3
 ↓      ↓      ↓
Tree1  Tree2  Tree3
```

Each tree then performs its normal Decision Tree training:

```text
Parent impurity
      ↓
Possible feature + thresholds
      ↓
Left / Right
      ↓
Weighted child impurity
      ↓
Gain
      ↓
Best split
      ↓
Repeat
```

So Random Forest **doesn't replace the Decision Tree algorithm**. It uses **many Decision Trees**.

---

## Q10. How does each tree select the best split?

Exactly like a Decision Tree.

Suppose Tree 1 has:

```text
Study_Hours
Previous_Marks
Attendance
```

It checks possible features and thresholds:

```text
Study_Hours <= 5.5
Previous_Marks <= 65
Attendance <= 80
```

For each candidate:

```text
Left / Right
     ↓
Left impurity
Right impurity
     ↓
Weighted child impurity
     ↓
Gain
```

Then:

```text
Lowest weighted child impurity
        ↓
Largest gain
        ↓
Best split
```

This is the same Decision Tree process: parent impurity → thresholds → left/right → weighted child impurity → gain → best split.

---

## Q11. What is `max_features`?

> Number of features considered when finding a split at each node.

Suppose we have 6 features:

```text
F1  F2  F3  F4  F5  F6
```

And:

```python
max_features=3
```

At one node:

```text
6 features
    ↓
Randomly select 3
    ↓
F1, F3, F5
    ↓
Check thresholds
    ↓
Calculate impurity/gain
    ↓
Choose best split
```

At another node, it can select another subset:

```text
F2, F4, F6
```

This gives different trees more diversity.

---

## Q12. What is `max_depth`?

> Maximum depth each Decision Tree can grow.

```python
max_depth=3
```

```text
Root
 ↓
Depth 1
 ↓
Depth 2
 ↓
Depth 3
 ↓
STOP
```

It prevents trees from growing indefinitely.

---

## Q13. What is `min_samples_split`?

> Minimum number of samples required in a node before that node can split.

```python
min_samples_split=5
```

```text
Node = 8 samples
8 >= 5 → can split ✅

Node = 4 samples
4 < 5 → cannot split ❌
```

So it checks the **current node** before splitting.

---

## Q14. What is `min_samples_leaf`?

> Minimum number of samples that must remain in each leaf after a split.

```python
min_samples_leaf=3
```

Suppose a possible split creates:

```text
8 / 2
```

Rejected:

```text
2 < 3 ❌
```

Try another threshold:

```text
6 / 4
```

Accepted:

```text
6 >= 3 ✅
4 >= 3 ✅
```

So:

```text
min_samples_split
→ Can the current node split?

min_samples_leaf
→ Are both resulting children large enough?
```

---

## Q15. What happens after all trees are trained?

```text
Tree 1 → complete
Tree 2 → complete
Tree 3 → complete
```

Now we have a trained Random Forest.

Then test data comes in:

```text
X_test
  ↓
Tree 1 → prediction
Tree 2 → prediction
Tree 3 → prediction
  ↓
Average
  ↓
Final prediction
```

---

## Q16. What does `model.predict()` do?

After training:

```python
model.fit(X_train, y_train)
```

we use:

```python
y_pred = model.predict(X_test)
```

It **does not train again**.

Test student:

```text
Study_Hours = 6.5
```

The same input goes through every trained tree:

```text
6.5
 ↓
Tree 1 → learned conditions → Leaf → 58
6.5
 ↓
Tree 2 → learned conditions → Leaf → 61
6.5
 ↓
Tree 3 → learned conditions → Leaf → 57
```

Then:

```text
Final = (58 + 61 + 57) / 3 = 58.67
```

---

## Q17. How is the final Random Forest prediction calculated?

For regression:

```text
ŷ_RF = (1/N) Σ ŷᵢ
```

Where:

- N = number of trees
- ŷᵢ = prediction from each tree

Example:

```text
Tree 1 → 58
Tree 2 → 61
Tree 3 → 57
```

```text
(58 + 61 + 57) / 3 = 58.67
```

**58.67 is the final Random Forest prediction.**

---

## Q18. What does a leaf predict?

Each Decision Tree in the Random Forest behaves like a regression tree.

At a leaf, prediction is normally the **mean target value of the training samples that reached that leaf**.

```text
Leaf:
61
72
79
```

Prediction:

```text
(61 + 72 + 79) / 3 = 70.67
```

```text
Leaf prediction = 70.67
```

---

## Q19. Does test data participate in training?

**No.**

```text
X_train + y_train
        ↓
      fit()
        ↓
Build all trees
```

Then:

```text
X_test
  ↓
predict()
  ↓
y_pred
  ↓
Compare with y_test
  ↓
Metrics
```

Remember:

```text
TRAIN → BUILD
TEST  → CHECK
```

---

## Q20. What are the important Random Forest hyperparameters?

```text
n_estimators
→ How many trees?

max_depth
→ How deep can each tree grow?

min_samples_split
→ How many samples are needed before a node can split?

min_samples_leaf
→ How many samples must remain in each leaf?

max_features
→ How many features are considered at each split?
```

---

## Q21. How does Random Forest reduce overfitting?

Random Forest uses **multiple sources of randomness**:

```text
Bootstrap sampling
       +
Random feature selection
       ↓
Different Decision Trees
       ↓
Combine their predictions
       ↓
More stable model
```

Instead of:

```text
One Tree → 60
```

we have:

```text
Tree 1 → 58
Tree 2 → 62
Tree 3 → 61
Tree 4 → 59
Tree 5 → 60
        ↓
Average → 60
```

One unusual tree has less influence on the final result because many trees contribute.

---

## Q22. What is overfitting in Random Forest?

```text
Training error = 2
Test error     = 20
```

```text
Training → very good
Test      → poor
```

This indicates possible overfitting.

We can tune:

```text
max_depth
min_samples_split
min_samples_leaf
max_features
```

to control model complexity and tree diversity.

---

## Q23. What is underfitting?

```text
Training error = 18
Test error     = 20
```

Both errors are high.

```text
Training → poor
Test      → poor
```

The model may be too simple.

---

## Q24. How do we evaluate Random Forest Regression?

We use:

```text
MAE
MSE
RMSE
R²
```

Example:

```text
Actual:
80, 90, 70

Predicted:
78, 87, 73
```

General interpretation:

```text
MAE  ↓ better
MSE  ↓ better
RMSE ↓ better
R²   ↑ better
```

---

## Q25. Practical Random Forest code

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
)

df = pd.read_csv("../data/students.csv")

X = df[["Study_Hours", "Previous_Marks", "Attendance_Percent"]]
y = df["Final_Marks"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2:", r2)
```

---

## Q26. Advantages of Random Forest

- Handles nonlinear relationships
- Handles multiple features
- Usually more stable than a single Decision Tree
- Does not normally require feature scaling
- Reduces the effect of one individual tree
- Works well for many tabular-data problems

---

## Q27. Disadvantages of Random Forest

- More computationally expensive than one Decision Tree
- Less interpretable than a single Decision Tree
- Many trees can require more memory
- Training and prediction can take longer
- Too much tuning can increase complexity

---

## Q28. Decision Tree vs Random Forest

```text
Decision Tree
→ One tree
→ Easy to understand
→ Can overfit easily
→ Faster / simple

Random Forest
→ Many trees
→ More stable
→ Usually better generalization
→ More computationally expensive
```

Example:

```text
Decision Tree:

Data
 ↓
One tree
 ↓
Prediction = 80
```

```text
Random Forest:

Data
 ↓
Tree 1 → 78
Tree 2 → 82
Tree 3 → 81
Tree 4 → 79
 ↓
Average → 80
```

---

## Q29. When would you choose Decision Tree instead of Random Forest?

> I would choose a Decision Tree when I need a simple, lightweight and interpretable model, especially when I want to understand the decision rules directly.

Example:

```text
Study_Hours > 6?
      ↓
    YES
      ↓
Attendance > 80?
      ↓
    YES
      ↓
Predicted Marks = 85
```

---

## Q30. When would you choose Random Forest?

> I would choose Random Forest when I want a more stable and generalized tree-based model and have enough computational resources to train multiple trees.

Example:

```text
Many features
      +
Complex nonlinear relationships
      ↓
Multiple Decision Trees
      ↓
Average predictions
      ↓
Random Forest
```

---

## Q31. Complete Random Forest workflow

This is the **most important flow to remember**:

```text
Dataset
   ↓
X features + y target
   ↓
Train / Test Split
   ↓
X_train + y_train
   ↓
Bootstrap Sampling
   ↓
Multiple training samples
   ↓
Build Tree 1, Tree 2, Tree 3...
   ↓
For each tree:
   Parent impurity
        ↓
   Possible features + thresholds
        ↓
   Left / Right
        ↓
   Weighted child impurity
        ↓
   Gain
        ↓
   Best split
        ↓
   Repeat recursively
        ↓
   Complete tree
   ↓
All trees trained
   ↓
X_test
   ↓
model.predict()
   ↓
Every tree gives prediction
   ↓
Average predictions
   ↓
Final Random Forest prediction
   ↓
Compare with y_test
   ↓
MAE / MSE / RMSE / R²
```

---

## Q32. Interview: "Explain Random Forest Regression"

Strong short answer:

> **Random Forest Regression is an ensemble supervised learning algorithm that combines multiple Decision Trees to predict a continuous target. During `fit()`, it creates different bootstrap samples from the training data and builds a Decision Tree for each sample. Each tree finds its best splits using feature thresholds, impurity and gain, while Random Forest also introduces feature randomness. After all trees are trained, `predict()` sends a new sample through every tree. Each tree produces a prediction, and for regression these predictions are averaged to produce the final Random Forest prediction.**

---

## Q33. One-line interview answers

**What is Random Forest?**  
A supervised ensemble algorithm that combines multiple Decision Trees.

**What is `n_estimators`?**  
Number of Decision Trees in the forest.

**What is Bootstrap Sampling?**  
Randomly sampling training rows with replacement to create different training samples.

**Why multiple trees?**  
To combine different tree predictions and obtain a more stable model.

**What is `max_depth`?**  
Maximum depth each tree can grow.

**What is `min_samples_split`?**  
Minimum samples required in a node before it can split.

**What is `min_samples_leaf`?**  
Minimum samples required in each resulting leaf.

**What is `max_features`?**  
Number of features considered at each split.

**What happens in `fit()`?**  
Bootstrap samples are created and multiple Decision Trees are trained.

**What happens in `predict()`?**  
The new sample passes through every trained tree and each tree produces a prediction.

**How is the final regression prediction calculated?**  
By averaging the predictions from all trees.

**Does `predict()` train again?**  
No. It only uses the already-trained trees.

**Does test data build the trees?**  
No. Training data builds the trees; test data evaluates them.

**How do we reduce overfitting?**  
Tune parameters such as `max_depth`, `min_samples_split`, `min_samples_leaf`, and `max_features`.

**Decision Tree vs Random Forest?**  
Decision Tree uses one tree; Random Forest combines many trees.

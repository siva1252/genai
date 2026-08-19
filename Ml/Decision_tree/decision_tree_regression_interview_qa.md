# Decision Tree Regression — Interview Q&A

Interview-ready notes covering **what → why → how → formulas → practical code → interview answer**.

---

## Q1. What is Decision Tree Regression?

**Decision Tree Regression is a supervised learning algorithm used to predict a continuous/numeric target by repeatedly splitting the training data based on feature conditions and making a numeric prediction at the leaf nodes.**

Example:

```text
Study_Hours
Previous_Marks
Attendance
       ↓
   Final_Marks
```

---

## Q2. Why do we use Decision Tree Regression?

Use it when:

- The target is **numeric / continuous**
- The relationship may be **nonlinear**
- You want a model that is **easy to interpret** as if/else rules
- You want a model that can split data into groups and predict a number in each group

---

## Q3. What happens in `model.fit()`?

```python
model.fit(X_train, y_train)
```

This is the **training process**. It builds the tree from training data only.

```text
Training data
      ↓
Find target average
      ↓
Calculate Parent Impurity
      ↓
Find possible feature thresholds
      ↓
Try every possible split
      ↓
Left / Right
      ↓
Calculate Left impurity
Calculate Right impurity
      ↓
Calculate Weighted Child Impurity
      ↓
Calculate Gain
      ↓
Choose BEST split
      ↓
Repeat recursively
      ↓
Create final tree
```

---

## Q4. What is the complete split-selection flow?

```text
1. Training target values
        ↓
2. Calculate target average
        ↓
3. Calculate Parent Impurity

I(node) = (1/n) Σ(yᵢ - ȳ)²
        ↓
4. Look at feature values
        ↓
5. Generate possible thresholds
        ↓
6. For each threshold:
        ↓
   Split → Left / Right
        ↓
7. Calculate average target
   for Left and Right
        ↓
8. Calculate impurity
   I(left), I(right)
        ↓
9. Calculate weighted child impurity

I(split) =
(n_left/n) × I(left)
+
(n_right/n) × I(right)

        ↓
10. Compare all candidate splits
        ↓
11. Smallest weighted child impurity
        ↓
12. Best split
        ↓
13. Calculate Gain

Gain = Parent Impurity - Weighted Child Impurity
        ↓
14. Larger Gain = better split
        ↓
15. Create Left / Right branches
        ↓
16. Repeat the same process
    inside the child nodes
```

---

## Q5. What is calculated first: target average?

Yes. For regression, the tree first looks at the **target/label**, not the feature average.

Example Marks:

```text
18, 35, 44, 53, 61, 72, 79, 95
```

Average:

```text
ȳ = Σy / n
ȳ = 57.125
```

---

## Q6. What is Parent Impurity?

Using the default `squared_error` criterion:

```text
I(node) = (1/n) Σ(yᵢ - ȳ)²
```

This tells how spread out the target values are **before splitting**.

Example:

```text
Parent Impurity = 544.859375
```

High impurity means the target values in that node are mixed / spread out.

---

## Q7. How does the tree find possible thresholds?

It looks at sorted feature values and tests boundaries between neighboring values.

Feature:

```text
Study_Hours:
1, 3, 4, 5, 6, 7, 8, 10
```

Possible thresholds:

```text
2
3.5
4.5
5.5
6.5
7.5
9
```

The tree tests these possible splits.

---

## Q8. How does a split look?

Suppose it tests:

```text
Study_Hours <= 5.5
```

Then:

```text
             <= 5.5?
            /       \
         YES         NO
          ↓           ↓
    18,35,44,53   61,72,79,95
```

Left = samples that satisfy the condition.  
Right = samples that do not.

---

## Q9. How is Left and Right impurity calculated?

Same formula as parent impurity, but using only the samples in that child.

Left:

```text
18, 35, 44, 53
I(left) = (1/n) Σ(yᵢ - ȳ)²
```

Right:

```text
61, 72, 79, 95
I(right) = (1/n) Σ(yᵢ - ȳ)²
```

Each side first calculates its own average, then its own impurity.

---

## Q10. What is Weighted Child Impurity?

This is the important formula:

```text
I(split) = (n_left / n) × I(left) + (n_right / n) × I(right)
```

Example:

```text
I(split) = 159.71875
```

This is the impurity **after that particular split**.

Larger groups get more weight because they contain more samples.

---

## Q11. What is Gain?

```text
Gain = Parent Impurity − Weighted Child Impurity
```

Example:

```text
Gain = 544.859375 − 159.71875
     = 385.140625
```

Meaning:

```text
Before split  →  544.859
After split   →  159.719
Reduction     →  385.141
```

Larger Gain = better split.

---

## Q12. How does it choose the best split?

It tests all candidate thresholds and compares the resulting weighted child impurity.

```text
Split A  →  Child impurity = 300
Split B  →  Child impurity = 200
Split C  →  Child impurity = 100
```

The tree chooses **Split C**.

Reason:

```text
Smaller resulting child impurity
              ↓
Larger impurity reduction
              ↓
Better split
```

Short interview line:

> Choose the split with the lowest resulting weighted impurity.

---

## Q13. What happens after the best split is chosen?

The process repeats inside the child nodes. This is **recursive splitting**.

```text
                Best Split
                 /       \
              Left       Right
               ↓           ↓
          Find again    Find again
          best split    best split
               ↓           ↓
             split       split
```

---

## Q14. When does the tree stop?

The tree stops according to conditions such as:   #overfiting we use this thing Decision Tree
→ max_depth
→ min_samples_split
→ min_samples_leaf

```python
max_depth  = depth of root 
min_samples_split
min_samples_leaf
```

model = DecisionTreeRegressor(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

max_depth
→ Maximum depth of the tree

min_samples_split
→ Minimum number of samples required to split a node

min_samples_leaf
→ Minimum number of samples required in a leaf


max_depth          → How deep the tree can go
min_samples_split  → How many samples are needed to split
min_samples_leaf   → How many samples must stay in each leaf


2. max_depth

Definition: Maximum depth each Decision Tree can grow.
Example: max_depth=5 → each tree can grow up to depth 5.

3. min_samples_split

Definition: Minimum number of samples required in a node before it can split.
Example: min_samples_split=5 → a node needs at least 5 samples to split.

4. min_samples_leaf

Definition: Minimum number of samples that must remain in each leaf after a split.
Example: min_samples_leaf=3 → both resulting leaves must have at least 3 samples.

5. max_features

Definition: Number of features considered when finding a split at each node.
Example: max_features=3 → if there are 6 features, 3 are considered at that split.




Then the remaining nodes become **leaf nodes**.

Without stopping rules, the tree can keep splitting until each leaf has very few samples and overfit.

---

## Q15. What does a leaf predict?

For regression, the leaf prediction is normally the **mean target value of the training samples reaching that leaf**.

Example:

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

## Q16. What does `predict()` do?

After the tree is trained:

```python
y_pred = model.predict(X_test)
```

The new sample travels through the learned conditions:

```text
New sample
    ↓
Condition
    ↓
YES / NO
    ↓
Condition
    ↓
YES / NO
    ↓
Leaf
    ↓
Prediction
```

It **does not train again**. It only follows the already-built tree.

---

## Q17. Is test data used to choose splits?

**No.**

Test data is **not** used to calculate parent impurity or choose splits.

```text
X_train + y_train
        ↓
     fit()
        ↓
   Build tree
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
MAE / MSE / RMSE / R²
```

Remember:

```text
TRAIN → BUILD
TEST  → CHECK
```

---

## Q18. What are the main formulas?

**Target mean**

```text
ȳ = Σy / n
```

**Node impurity**

```text
I(node) = (1/n) Σ(yᵢ - ȳ)²
```

**Weighted child impurity**

```text
I(split) = (n_left / n) × I(left) + (n_right / n) × I(right)
```

**Gain**

```text
Gain = Parent Impurity − Weighted Child Impurity
```

**Leaf prediction**

```text
Prediction = Mean(target values in leaf)
```

---

## Q19. Decision Tree vs Linear Regression

**Linear Regression**

```text
Training data
     ↓
OLS
     ↓
Coefficients + Intercept
     ↓
Equation
     ↓
Best-fit line
```

**Decision Tree Regression**

```text
Training data
     ↓
Target average
     ↓
Parent impurity
     ↓
Possible thresholds
     ↓
Left / Right
     ↓
Child impurity
     ↓
Weighted child impurity
     ↓
Gain
     ↓
Best split
     ↓
Repeat
     ↓
Tree + Leaves
```

Linear Regression learns one equation.  
Decision Tree learns a set of if/else splits.

---

## Q20. What is `criterion="squared_error"`?

Default for `DecisionTreeRegressor`.

It means node impurity is measured as:

```text
I(node) = (1/n) Σ(yᵢ - ȳ)²
```

This is the same idea as MSE inside a node.

```python
model = DecisionTreeRegressor(criterion="squared_error")
```

---

## Q21. Why train/test split?

Same reason as Linear Regression.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

Training data is used to **build** the tree.  
Test data is used to **evaluate** predictions.

---

## Q22. Practical training and prediction code

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score

X = df[["Study_Hours", "Previous_Marks", "Attendance_Percent"]]
y = df["Final_Marks"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

Then evaluate:

```python
mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)
```

---

## Q23. Why do we still use MAE, MSE, RMSE and R²?

Because building a tree is not enough. We must check how well it predicts **unseen** data.

| Metric | What it tells |
|--------|----------------|
| MAE    | Average absolute prediction error |
| MSE    | Average squared error; large errors are penalized more |
| RMSE   | Error in original target units |
| R²     | How much target variation the model explains |

```text
MAE    ↓  better
MSE    ↓  better
RMSE   ↓  better
R²     ↑  better
```

---

## Q24. Advantages

- Easy to interpret as if/else rules
- Captures **nonlinear** relationships
- No need for a linear equation
- Can handle multiple features
- Feature scaling is usually not required
- Good for understanding which feature conditions matter

---

## Q25. Disadvantages

- Can **overfit** easily if the tree grows too deep
- Sensitive to small changes in data
- Predictions are stepwise, not a smooth line
- May be less stable than Linear Regression or Random Forest
- Deep trees can become hard to interpret

---

## Q26. When would you choose Decision Tree Regression?

> I would consider Decision Tree Regression when the target is continuous and the relationship may be nonlinear or based on feature conditions. I would train it on unseen-holdout data, control overfitting with parameters such as `max_depth`, evaluate with MAE, MSE, RMSE and R², and compare it with other models such as Linear Regression.

---

## Q27. Interview: "Explain how Decision Tree Regression works."

Strong short answer:

> Decision Tree Regression is a supervised learning algorithm used to predict continuous values. During `fit()`, the algorithm calculates the impurity of the current node, considers possible feature thresholds, and evaluates each split by calculating the weighted impurity of the resulting child nodes. It chooses the split with the lowest weighted child impurity, which is equivalent to the largest impurity reduction. It recursively repeats this process on the child nodes until stopping conditions are reached. Finally, the leaf nodes make numeric predictions, typically using the mean target value of the samples in that leaf.

---

## Q28. One-line interview answers

**What is Decision Tree Regression?**  
A supervised algorithm that predicts a continuous target by splitting data on feature conditions and predicting the mean target value at each leaf.

**What is `fit()`?**  
The training step that builds the tree by finding the best splits from training data.

**What is parent impurity?**  
How spread out the target values are in a node before splitting.

**What is a threshold?**  
A candidate feature cutoff used to send samples left or right.

**What is weighted child impurity?**  
The combined left and right impurity after a split, weighted by sample size.

**What is Gain?**  
Parent impurity minus weighted child impurity. Larger Gain means a better split.

**How is the best split chosen?**  
The split with the lowest weighted child impurity, which also gives the largest Gain.

**What does a leaf predict?**  
The average target value of the training samples in that leaf.

**What is `predict()`?**  
It sends a new sample down the learned conditions until a leaf, then returns that leaf’s mean.

**Is test data used to build the tree?**  
No. Train builds the tree. Test only evaluates it.

**When does splitting stop?**  
When stopping rules such as `max_depth`, `min_samples_split`, or `min_samples_leaf` are reached.

**Decision Tree vs Linear Regression?**  
Linear Regression learns one equation. Decision Tree learns recursive if/else splits.





1. Training target values
        ↓
2. Calculate target average
        ↓
3. Calculate Parent Impurity

I(node) = (1/n) Σ(yᵢ - ȳ)²
        ↓
4. Look at feature values
        ↓
5. Generate possible thresholds
        ↓
6. For each threshold:
        ↓
   Split → Left / Right
        ↓
7. Calculate average target
   for Left and Right
        ↓
8. Calculate impurity
   I(left), I(right)
        ↓
9. Calculate weighted child impurity

I(split) =
(n_left/n) × I(left)
+
(n_right/n) × I(right)

        ↓
10. Compare all candidate splits
        ↓
11. Smallest weighted child impurity
        ↓
12. Best split
        ↓
13. Calculate Gain

Gain = Parent Impurity - Weighted Child Impurity
        ↓
14. Larger Gain = better split
        ↓
15. Create Left / Right branches
        ↓
16. Repeat the same process
   inside the child nodes
# Linear Regression — Interview Q&A

Interview-ready notes covering **concepts + why + how + formulas + practical usage**.


Linear Regression
→ Find coefficients/intercept
→ Minimize squared prediction error

---

## Q1. What is Linear Regression?

**Linear Regression is a supervised machine learning algorithm used to predict a continuous/numeric target by learning the linear relationship between input features and the target.**

Examples:

```text
Study Hours  →  Marks
Experience   →  Salary
Area         →  House Price
```

---

## Q2. Why do we use Linear Regression?

Use it when:

- The target is **numeric / continuous**
- You want to predict a numerical value
- There is a reasonably **linear relationship** between features and target
- You want a **simple and interpretable** model

---

## Q3. What is the basic formula?

**One feature (Simple Linear Regression):**

```text
ŷ = b₀ + b₁x
```

| Symbol | Meaning |
|--------|---------|
| ŷ      | Predicted value |
| b₀     | Intercept |
| b₁,b2,b3    | Coefficient |
| x      | Feature |

**Multiple features (Multiple Linear Regression):**

```text
ŷ = b₀ + b₁x₁ + b₂x₂ + ... + bₙxₙ
```

---
in overfiitng we are use thsi things 
Linear Regression
→ Regularization
→ Ridge / Lasso / Elastic Net

## Q4. What is Simple Linear Regression?

When we have **1 feature + 1 target**.

Example:

```text
Study_Hours  →  Marks
```

Formula:

```text
ŷ = b₀ + b₁x
```

---

## Q5. What is Multiple Linear Regression?

When we have **multiple features + 1 target**.

Example:

```text
Study_Hours
Previous_Marks
Attendance
        ↓
      Marks
```

Formula:

```text
ŷ = b₀ + b₁x₁ + b₂x₂ + b₃x₃
```

---

## Q6. What is a feature?

A **feature is an input variable used by the model to make a prediction.**

Example:

```text
Study_Hours
Attendance
Previous_Marks
```

---

## Q7. What is a label / target?

The **label/target is the value we want the model to predict.**

Example:

```text
Features  →  Study_Hours, Attendance
Target    →  Marks
```

---

## Q8. Why is X uppercase and y lowercase?

Convention:

```python
X = features   # can have multiple columns
y = target     # normally one column
```

```python
X = df[["Study_Hours", "Attendance"]]
y = df["Marks"]
```

`X` is uppercase because it is typically a **matrix** (many samples × many features).  
`y` is lowercase because it is typically a **vector** (one target per sample).

---

## Q9. Why is X 2D?

Scikit-learn expects features in this shape:

```text
(samples, features)
```

Even with **one feature**, pass a 2D matrix:

```python
X = df[["Study_Hours"]]   # 2D  — correct
```

Not:

```python
X = df["Study_Hours"]     # 1D  — wrong for sklearn
```

---

## Q10. Why do we split train and test data?

To evaluate the model on **data it did not train on**.

```text
Dataset
   ↓
Train / Test Split
   ↓
80% Training   →  learn
20% Testing    →  evaluate
```

If you test on the same data used for training, you cannot tell whether the model generalizes.

---

## Q11. What is `test_size=0.2`?

It means approximately:

```text
80%  →  training
20%  →  testing
```

`0.2` is common, not mandatory. The split depends on dataset size and the problem.

---

## Q12. What is `random_state=42`?

It makes the random train/test split **repeatable**.

The number `42` has **no special mathematical meaning**. Any fixed integer works. Using the same value gives the same split every run.

---

## Q13. What is `model.fit()`?

This is the **training step**.

```python
model.fit(X_train, y_train)
```

It learns the relationship:

```text
X_train  →  y_train
```

For Linear Regression it learns:

- Intercept (`b₀`)
- Coefficients (`b₁, b₂, ...`)

---

## Q14. How does `fit()` find coefficients?

Linear Regression commonly uses **Ordinary Least Squares (OLS)**.

Closed-form solution (multiple linear regression):

```text
β = (XᵀX)⁻¹ Xᵀ y
```

Objective: find coefficients that minimize the sum of squared errors:

```text
Σ (Actual − Predicted)²
```

You do not calculate this by hand in normal Scikit-learn usage. `model.fit(X_train, y_train)` does it internally.

---

## Q15. What is the intercept?

The intercept `b₀` is the **predicted target when all input features are zero**.

```text
ŷ = b₀ + b₁x
```

| Symbol | Meaning |
|--------|---------|
| ŷ      | Predicted label |
| y      | Actual label |
| x      | Feature / input |
| b₀     | Intercept |
| b₁     | Coefficient |

---

## Q16. What is a coefficient?

A coefficient tells **how the predicted target changes when that feature increases by one unit, while other features are held constant**.

Example:

```text
Study_Hours coefficient = 8.66
```

If Study Hours increases by 1, predicted Marks increase by about **8.66**, assuming other features stay constant.

---

## Q17. What is `model.predict()`?

After training:

```python
y_pred = model.predict(X_test)
```

It uses the learned intercept and coefficients to produce predictions.

```text
X_test
   ↓
Trained model
   ↓
y_pred
```

---

## Q18. What happens with new data?

We **do not train again**. We reuse the trained model.

```text
Trained model
      ↓
New features
      ↓
model.predict()
      ↓
New prediction
```

Example:

```python
model.predict([[6.5]])
```

The value `6.5` would normally come from an application, API, database, or user input.

---

## Q19. What is residual / error?

For each prediction:

```text
Error = Actual − Predicted
```

Example:

```text
Actual     = 88
Predicted  = 87.45

Error = 88 − 87.45 = 0.55
```

---

## Q20. Why do we need metrics?

Producing predictions is not enough. We need to know:

> How well is the model performing on unseen test data?

That is why we use regression metrics: **MAE, MSE, RMSE, R²**.

---

## Q21. What is MAE?

**Mean Absolute Error**

```text
MAE = (1/n) Σ |y − ŷ|
```

- Average size of the prediction error
- **Lower is better**

Example: `MAE = 2` means predictions are off by about **2 units** on average.

---

## Q22. What is MSE?

**Mean Squared Error**

```text
MSE = (1/n) Σ (y − ŷ)²
```

- Average of **squared** prediction errors
- Large errors are penalized more because they are squared
- **Lower is better**

---

## Q23. What is RMSE?

**Root Mean Squared Error**

```text
RMSE = √MSE
```

- Error size in the **original target unit**
- More sensitive to large errors than MAE
- **Lower is better**

---

## Q24. What is R²?

**R-squared**

```text
R² = 1 − [ Σ(y − ŷ)²  /  Σ(y − ȳ)² ]
```

- How much variation in the target is explained by the model compared with an **average-only baseline**
- **Higher is generally better**
- **R² is not prediction accuracy**

---

## Q25. Why use all four metrics?

They answer different questions.

| Metric | Question it answers |
|--------|---------------------|
| MAE    | How far are predictions wrong on average? |
| MSE    | How large are the squared errors? |
| RMSE   | How large are errors in original units? |
| R²     | How much target variation does the model explain? |

---

## Q26. How do we compare algorithms?

Train several models on the **same dataset and same test set**, then compare metrics.

Example:

```text
                 MAE     RMSE     R²
Linear           2.5      3.0    0.90
Decision Tree    1.8      2.2    0.94
Random Forest    1.2      1.6    0.97
```

Rule of thumb:

```text
MAE    ↓  better
MSE    ↓  better
RMSE   ↓  better
R²     ↑  better
```

Choose the model based on **performance + business requirement**, not one metric alone.

---

## Q27. What are the assumptions of Linear Regression?

Very common interview question.

1. **Linearity** — relationship between features and target is approximately linear
2. **Independence** — observations / errors should be reasonably independent
3. **Homoscedasticity** — error variance should be reasonably constant
4. **Low multicollinearity** — predictors should not be excessively correlated with each other
5. **Residual normality** — for statistical inference, residuals are often assumed to be approximately normally distributed

---

## Q28. What are the advantages of Linear Regression?

- Simple
- Fast to train
- Easy to interpret
- Coefficients are understandable
- Good **baseline** model
- Works well when relationships are approximately linear

---

## Q29. What are the disadvantages of Linear Regression?

- Struggles with strongly **nonlinear** relationships
- Sensitive to **outliers**
- **Multicollinearity** can make coefficients unstable
- May underperform more flexible models on complex datasets

---

## Q30. When would you choose Linear Regression?

> I would consider Linear Regression when the target is continuous and the relationship between the predictors and the target is reasonably linear. I would train it, evaluate it on unseen data using MAE, MSE, RMSE and R², and compare it with other suitable models before choosing.

---

## Q31. Interview: "Explain Linear Regression"

Strong short answer:

> Linear Regression is a supervised learning algorithm used to predict a continuous numerical target. It learns a linear relationship between input features and the target. For a single feature the equation is `ŷ = b₀ + b₁x`; for multiple features it is `ŷ = b₀ + b₁x₁ + ... + bₙxₙ`. During training, the model learns the coefficients using Ordinary Least Squares by minimizing the sum of squared errors. After training we use `predict()` on unseen data and evaluate with MAE, MSE, RMSE and R².

---

## Q32. One-line interview answers

**What is Linear Regression?**  
A supervised learning algorithm for predicting continuous numerical values using a linear relationship between features and target.

**What is `fit()`?**  
The training operation that learns the model parameters from training data.

**What is `predict()`?**  
It uses the learned parameters to generate predictions for new feature values.

**What is the coefficient?**  
The change in predicted target for a one-unit change in a feature, holding other features constant.

**What is the intercept?**  
The predicted target when all features are zero.

**Why train/test split?**  
To evaluate the model on unseen data.

**Why MAE?**  
To measure average absolute prediction error.

**Why MSE?**  
To measure squared error and penalize large errors more heavily.

**Why RMSE?**  
To measure error in the original target unit while remaining sensitive to large errors.

**Why R²?**  
To measure how much variation in the target is explained by the model relative to an average-only baseline.

**Which metric is better?**  
MAE, MSE and RMSE are lower-is-better; R² is generally higher-is-better.

**How do you select the best regression model?**  
Evaluate candidate models on unseen data using appropriate metrics and select the model that best satisfies both the evaluation results and the business objective.




1. Business problem
        ↓
2. Dataset
        ↓
3. Identify Features (X) and Label (y)
        ↓
4. Since label is numeric
        ↓
   Supervised Learning
        ↓
   Regression
        ↓
5. Split dataset
        ↓
   X_train, X_test
   y_train, y_test
        ↓
6. Choose a regression algorithm
   Example: Linear Regression
        ↓
7. model.fit(X_train, y_train)
        ↓
   Model learns from TRAINING data
        ↓
   Learns coefficients/intercept
        ↓
8. model.predict(X_test)
        ↓
   Get predicted values
        ↓
9. Compare:
   y_test vs y_pred
        ↓
10. Evaluation metrics
    MAE
    MSE
    RMSE
    R²
        ↓
11. Try other suitable regression algorithms
        ↓
12. Compare their test performance
        ↓
13. Select the model that fits the problem best
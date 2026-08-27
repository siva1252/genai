# Logistic Regression — Interview & Implementation Notes

Core flow:

```text
Dataset
  → X / y
  → train/test split
  → scaling when appropriate
  → model.fit()
  → learned coefficients / intercept
  → z score
  → sigmoid
  → probability
  → threshold
  → predicted class
  → confusion matrix
  → Accuracy / Precision / Recall / F1
  → ROC-AUC
```

---

## 1. What is Logistic Regression?

Logistic Regression is a supervised learning algorithm mainly used for **binary classification**.

It:

1. Calculates a linear score `z`
2. Converts `z` to a probability with the **sigmoid**
3. Uses a **decision threshold** to produce a class

**Interview answer:**  
Logistic Regression is a supervised classification algorithm that estimates class probabilities using a linear combination of features followed by the sigmoid function.

---

## 2. Simple and Multiple Features

**One feature:**

```text
z = w1 * x1 + b
```

**Multiple features:**

```text
z = w1*x1 + w2*x2 + … + wn*xn + b
```

Every feature has a learned coefficient; the rest of the process is unchanged.

In this project:

- Simple (`simply.py`): `Study_Hours` → `Result`
- Multiple (`multiple.py`): `Study_Hours`, `Previous_Marks`, `Attendance_Percent` → `Result`

---

## 3. w and b

- `w` = learned coefficients
- `b` = intercept

They are learned during `model.fit()`. You normally do **not** choose them manually.

In sklearn:

```python
model.coef_
model.intercept_
```

---

## 4. z

`z` is the linear score:

```text
z = w · x + b
```

- Positive `z` → probability toward **1**
- Negative `z` → probability toward **0**

A formula like `z = 1.5x − 6` is only an example for one trained model, **not** a fixed Logistic Regression formula.

---

## 5. Sigmoid

```text
σ(z) = 1 / (1 + e^(−z))
```

It maps any real `z` to a value between **0 and 1**, interpreted as the estimated probability of Class 1.

---

## 6. Threshold

A threshold converts probability to a class.

Common rule:

```text
probability ≥ 0.5  → Class 1
probability < 0.5  → Class 0
```

The threshold is a **decision rule**. You can change it when the problem needs a different Precision / Recall trade-off.

---

## 7. What happens inside `model.fit()`?

Conceptually:

```text
X_train + y_train
      ↓
Logistic Regression
      ↓
learns w and b
      ↓
calculates z = wx + b
      ↓
sigmoid → probability
      ↓
Log Loss
      ↓
optimization adjusts w, b
      ↓
repeat until convergence
      ↓
final / best w, b
```

Scikit-learn handles these calculations internally.  
With regularization enabled, the objective also includes a regularization term.

---

## 8. Log Loss

For one binary observation:

```text
Loss = −[ y log(p) + (1 − y) log(1 − p) ]
```

- Confident **correct** probabilities → low loss
- Confident **wrong** probabilities → high loss

Logistic Regression uses this objective, **not** ordinary linear-regression MSE, as its main classification loss.

---

## 9. Scaling

Scaling is often useful, especially with:

- regularization
- features on very different numerical ranges

Correct workflow:

```python
scaler.fit_transform(X_train)
scaler.transform(X_test)
```

Never fit the scaler separately on test data.

In this project, `multiple.py` uses `StandardScaler`.

---

## 10. Important Hyperparameters

| Parameter | Meaning |
|-----------|---------|
| `C` | Inverse regularization strength (smaller `C` = stronger regularization) |
| `penalty` | Regularization type |
| `solver` | Optimization algorithm |
| `max_iter` | Maximum iterations |
| `class_weight` | Useful for class imbalance |

The classification threshold is a **decision setting**, not the usual fit hyperparameter.

---

## 11. Prediction

```python
model.predict(X)        # predicted classes
model.predict_proba(X)  # probabilities
```

For binary classification:

```python
predict_proba(X)[:, 1]  # probability of Class 1
```

ROC-AUC needs probabilities, not only hard class labels.

---

## 12. Confusion Matrix

Compare actual and predicted classes:

| Actual | Predicted | Name | Meaning |
|-------:|----------:|------|---------|
| 0 | 0 | **TN** | Correctly predicted negative |
| 0 | 1 | **FP** | Negative predicted as positive |
| 1 | 0 | **FN** | Positive predicted as negative |
| 1 | 1 | **TP** | Correctly predicted positive |

Standard sklearn layout:

```text
[[TN, FP],
 [FN, TP]]
```

These four counts form Accuracy, Precision, Recall, and F1.

---

## 13. Metrics

```text
Accuracy  = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 × Precision × Recall / (Precision + Recall)
```

Memory aids:

- **Precision:** “When I predicted positive, was I right?”
- **Recall:** “Of all actual positives, how many did I find?”
- **F1:** balances Precision and Recall
- **Accuracy** can be misleading on highly imbalanced data

See also: `Ml/Classification/metrics.md`

---

## 14. ROC-AUC

ROC-AUC measures how well positive and negative examples are separated / ranked across thresholds.

- Uses **probability or score** outputs, not only final 0/1 predictions
- ROC plots **True Positive Rate** vs **False Positive Rate** at different thresholds
- AUC **1.0** = perfect ranking
- AUC **~0.5** = random-like

---

## 15. Overfitting and Underfitting

| Problem | Meaning |
|---------|---------|
| **Underfitting** | Too simple; poor on train and unseen data |
| **Overfitting** | Fits train very well; generalizes poorly |

Regularization and cross-validation help control these issues.

---

## 16. Model Selection

- Compare models on the **same** evaluation setup
- Use Accuracy, Precision, Recall, F1, and ROC-AUC according to the problem
- Inspect the confusion matrix
- Do **not** automatically pick highest Accuracy — FP / FN cost matters
- Prefer cross-validation on training data for model selection
- Keep the final test set untouched until final evaluation

---

## 17. Multiple-Feature Implementation

Concept (aligned with this project):

```python
X = data[["Study_Hours", "Previous_Marks", "Attendance_Percent"]]
y = data["Result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

print(model.coef_)
print(model.intercept_)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]
```

For multiple features:

```text
z = w1*x1 + w2*x2 + w3*x3 + b
```

Then:

```text
sigmoid → probability → threshold → class
```

The algorithm itself does not change.

---

## 18. Complete Metric Code

```python
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print(precision_score(y_test, y_pred))
print(recall_score(y_test, y_pred))
print(f1_score(y_test, y_pred))
print(roc_auc_score(y_test, y_prob))
```

---

## 19. Common Interview Questions

**Q: Why is it called Logistic Regression if it is classification?**  
A: The name comes from the logistic function / modeling formulation, but its common ML use is classification.

**Q: Why sigmoid?**  
A: It maps the linear score to a probability between 0 and 1.

**Q: How are w and b learned?**  
A: Optimization minimizes logistic loss, with regularization depending on configuration.

**Q: Why not Linear Regression for binary classes?**  
A: Linear Regression can produce values outside `[0, 1]` and does not use the appropriate probabilistic logistic objective.

**Q: Precision vs Recall?**  
A: Precision focuses on FP; Recall focuses on FN.

**Q: Why F1?**  
A: It gives a single balance between Precision and Recall.

**Q: Why ROC-AUC?**  
A: It evaluates probability-score separation across thresholds.

**Q: Why scale?**  
A: It can improve optimization and makes regularized coefficients more comparable across feature scales.

---

## 20. Final Interview Explanation

> Logistic Regression is a supervised classification algorithm. It calculates `z = w · x + b`, where the coefficients and intercept are learned during training by optimizing logistic loss, with regularization depending on the configuration. The sigmoid converts `z` into a probability between 0 and 1. A threshold converts that probability into a class. I evaluate predictions using a confusion matrix, Accuracy, Precision, Recall, F1-score and, when appropriate, ROC-AUC. With multiple features, `z` is simply the weighted sum of all features plus the intercept.

---

## Final Mental Model

```text
X
 → scaling when appropriate
 → z = w·x + b
 → sigmoid
 → probability
 → threshold
 → class
 → confusion matrix
 → Accuracy / Precision / Recall / F1

Probability scores → ROC-AUC
```

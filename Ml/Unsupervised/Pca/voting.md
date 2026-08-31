Yes mowa, **Voting Classifier is also part of Scikit-learn**. Let's learn it properly step by step.

# Voting Classifier

## 1. What is Voting Classifier?

Voting Classifier is an **ensemble learning technique**.

Instead of using only one classification model:

```text
Dataset
   ↓
One Model
   ↓
Prediction
```

we use **multiple different classification models**:

```text
             ┌→ Logistic Regression ─→ Class 1
Dataset ─────┼→ Decision Tree ───────→ Class 1
             └→ KNN ─────────────────→ Class 0
                         ↓
                    Voting
                         ↓
                      Class 1
```

The idea is:

> **Use multiple models and combine their predictions to make a final prediction.**

---

# 2. Why do we use Voting Classifier?

Suppose we have three models:

```text
Logistic Regression → 1
Decision Tree       → 1
KNN                 → 0
```

Two models predicted:

```text
1
```

One model predicted:

```text
0
```

So Voting Classifier chooses:

```text
Final prediction → 1
```

This can make the final prediction more robust than relying on one model.

---

# 3. Simple real-world example

Imagine three doctors are diagnosing a patient.

```text
Doctor 1 → Disease
Doctor 2 → Disease
Doctor 3 → No Disease
```

Majority says:

```text
Disease
```

Voting Classifier works similarly.

```text
Model 1 → Class A
Model 2 → Class A
Model 3 → Class B

Final → Class A
```

---

# 4. What models can we use?

We can combine different classifiers such as:

```text
Logistic Regression
Decision Tree
KNN
SVM
Random Forest
```

For example:

```text
Logistic Regression
Decision Tree
KNN
       ↓
Voting Classifier
       ↓
Final prediction
```

The models should ideally have **different strengths/errors** so that combining them is useful.

---

# 5. Hard Voting

There are two important types.

### Hard Voting

Each model gives a **class prediction**.

Example:

```text
Logistic Regression → Class 1
Decision Tree       → Class 1
KNN                 → Class 0
```

Voting:

```text
Class 1 → 2 votes
Class 0 → 1 vote
```

Final:

```text
Class 1
```

So:

> **Hard voting chooses the class with the majority of model predictions.**

---

# 6. Soft Voting

Soft voting uses **probabilities** instead of only class labels.

Suppose:

```text
Logistic Regression:

Class 0 → 0.30
Class 1 → 0.70
```

Decision Tree:

```text
Class 0 → 0.20
Class 1 → 0.80
```

KNN:

```text
Class 0 → 0.60
Class 1 → 0.40
```

Voting Classifier combines the probabilities.

Average probability for Class 0:

```text
(0.30 + 0.20 + 0.60) / 3
= 0.367
```

Class 1:

```text
(0.70 + 0.80 + 0.40) / 3
= 0.633
```

Therefore:

```text
Final → Class 1
```

So:

> **Soft voting combines the predicted probabilities from multiple classifiers and chooses the class with the highest combined probability.**

---

# 7. Hard vs Soft Voting

| Hard Voting                         | Soft Voting                       |
| ----------------------------------- | --------------------------------- |
| Uses class predictions              | Uses probabilities                |
| Majority vote                       | Combines probabilities            |
| `voting="hard"`                     | `voting="soft"`                   |
| Does not require probability output | Models need probability estimates |

---

# 8. Coding Example

Let's use the **Iris dataset**.

```python
import pandas as pd

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.metrics import accuracy_score, classification_report
```

Load data:

```python
iris = load_iris()

X = iris.data
y = iris.target
```

Split:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Scale the data:

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
```

Create individual models:

```python
model1 = LogisticRegression()

model2 = DecisionTreeClassifier(
    random_state=42
)

model3 = KNeighborsClassifier(
    n_neighbors=5
)
```

Create Voting Classifier:

```python
voting_model = VotingClassifier(
    estimators=[
        ('lr', model1),
        ('dt', model2),
        ('knn', model3)
    ],
    voting='hard'
)
```

Train:

```python
voting_model.fit(
    X_train_scaled,
    y_train
)
```

Predict:

```python
y_pred = voting_model.predict(
    X_test_scaled
)
```

Evaluate:

```python
accuracy = accuracy_score(
    y_test,
    y_pred
)

print("Accuracy:", accuracy)

print(
    classification_report(
        y_test,
        y_pred
    )
)
```

---

# 9. What happens inside `.fit()`?

When we execute:

```python
voting_model.fit(X_train_scaled, y_train)
```

conceptually:

```text
Training data
     ↓
 ┌───────────────┐
 ↓       ↓       ↓
LR      DT      KNN
 ↓       ↓       ↓
Train   Train   Train
 └───────┬───────┘
         ↓
Voting Classifier
```

The individual models are trained.

---

# 10. What happens inside `.predict()`?

Suppose a new customer comes:

```text
New Customer
     ↓
 ┌───────────────┐
 ↓       ↓       ↓
LR      DT      KNN
 ↓       ↓       ↓
  1       1       0
 └───────┬───────┘
         ↓
       Voting
         ↓
      Class 1
```

So:

```text
LR  → 1
DT  → 1
KNN → 0

Final → 1
```

This is why **Voting Classifier supports `predict()`**.

---

# 11. Soft Voting code

Change:

```python
voting='hard'
```

to:

```python
voting='soft'
```

Example:

```python
voting_model = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression()),
        ('dt', DecisionTreeClassifier(random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ],
    voting='soft'
)
```

Then:

```python
voting_model.fit(
    X_train_scaled,
    y_train
)

y_pred = voting_model.predict(
    X_test_scaled
)
```

The models provide probabilities, and Voting Classifier combines them.

---

# 12. What is `estimators`?

This:

```python
estimators=[
    ('lr', model1),
    ('dt', model2),
    ('knn', model3)
]
```

means:

```text
name → model
```

For example:

```text
lr  → Logistic Regression
dt  → Decision Tree
knn → KNN
```

---

# 13. Can we use the same model multiple times?

Technically, estimators can be specified, but using the same type repeatedly generally provides less diversity.

The main benefit of voting comes from combining models that make **different errors**.

---

# 14. Why can Voting improve performance?

Suppose:

```text
Model A → 90%
Model B → 88%
Model C → 87%
```

Each model makes some mistakes.

If their mistakes are different, combining them can produce a stronger final predictor.

The key idea is:

> **Diversity among models can make ensemble predictions more robust.**

But Voting Classifier **does not guarantee** better performance.

We should compare:

```text
Logistic Regression
vs
Decision Tree
vs
KNN
vs
Voting Classifier
```

using the same validation/test methodology.

---

# 15. Voting Classifier vs Random Forest

This is an important interview question.

### Voting Classifier

You explicitly combine different models:

```text
Logistic Regression
Decision Tree
KNN
       ↓
Voting
```

### Random Forest

Random Forest builds many **decision trees** and combines them.

```text
Decision Tree
Decision Tree
Decision Tree
Decision Tree
...
       ↓
Random Forest
```

So:

> **Voting can combine different types of classifiers, whereas Random Forest is an ensemble specifically made from decision trees.**

---

# 16. Voting vs Bagging

### Voting

Combines predictions from potentially different models.

```text
LR + DT + KNN
```

### Bagging

Usually trains multiple models of the same base type on different bootstrap samples and aggregates them.

Example:

```text
Tree
Tree
Tree
Tree
 ↓
Bagging
```

---

# 17. Voting vs Boosting

### Voting

Models are generally trained independently and their predictions are combined.

```text
Model 1 ──┐
Model 2 ──┼→ Voting
Model 3 ──┘
```

### Boosting

Models are trained sequentially, with later models focusing on errors made by earlier models.

```text
Model 1
   ↓
Model 2
   ↓
Model 3
   ↓
Final model
```

---

# 18. Is Voting Classifier supervised or unsupervised?

**Answer:**

> Voting Classifier is a supervised ensemble-learning technique because it combines supervised classification models and requires labeled training data.

---

# 19. Does Voting Classifier work for regression?

The `VotingClassifier` is specifically for **classification**.

For regression, scikit-learn provides:

```python
VotingRegressor
```

Example:

```text
VotingClassifier → Classification
VotingRegressor  → Regression
```

---

# 20. Interview: When would you use Voting Classifier?

**Answer:**

> I would consider Voting Classifier when I have multiple good classification models with complementary strengths and want to combine their predictions. I would evaluate whether the ensemble actually improves validation performance compared with the individual models.

---

# ⭐ Perfect interview answer

If the interviewer asks:

**"What is Voting Classifier?"**

Say:

> **"Voting Classifier is an ensemble-learning technique in Scikit-learn that combines multiple classification models and uses their predictions to make a final prediction. It supports hard voting, where the majority class prediction is selected, and soft voting, where predicted probabilities are combined and the class with the highest combined probability is selected. For example, I can combine Logistic Regression, Decision Tree and KNN. Each model makes a prediction for a new sample, and the Voting Classifier combines those predictions to produce the final class. The goal is to make the prediction more robust, although I would always compare its performance with the individual models."**

### The mental picture:

```text
              Training Data
                   ↓
       ┌───────────┼───────────┐
       ↓           ↓           ↓
 Logistic       Decision      KNN
Regression       Tree
       ↓           ↓           ↓
       └───────────┼───────────┘
                   ↓
              VOTING
                   ↓
           Final Prediction
```

And yes — **this is Scikit-learn ensemble learning**, so adding Voting Classifier to your ML preparation is correct.

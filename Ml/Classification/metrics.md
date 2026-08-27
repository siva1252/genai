# Classification Metrics — Complete Notes

After training a classification model, we compare:

```text
Actual values
      ↓
Predicted values
      ↓
Confusion Matrix
      ↓
TN, FP, FN, TP
      ↓
Accuracy
Precision
Recall
F1-score
```

ROC-AUC is slightly different because it uses **probability scores**.

---

# 1. Confusion Matrix

A confusion matrix tells us **what kind of predictions the model made**.

Suppose:

```text
1 = Pass
0 = Fail
```

We compare:

```text
Actual class
     vs
Predicted class
```

There are four possibilities:

| Actual | Predicted | Name   | Meaning                        |
| -----: | --------: | ------ | ------------------------------ |
|      0 |         0 | **TN** | Correctly predicted negative   |
|      0 |         1 | **FP** | Negative predicted as positive |
|      1 |         0 | **FN** | Positive predicted as negative |
|      1 |         1 | **TP** | Correctly predicted positive   |

The standard layout is:

```text
                 Predicted
                 0       1

Actual 0        TN      FP

Actual 1        FN      TP
```

### Example

```text
Actual:     [0, 0, 1, 0]
Predicted:  [0, 0, 1, 0]
```

Compare one by one:

```text
0 → 0 = TN
0 → 0 = TN
1 → 1 = TP
0 → 0 = TN
```

Therefore:

```text
TN = 3
FP = 0
FN = 0
TP = 1
```

Confusion matrix:

```text
[[3 0]
 [0 1]]
```

---

# 2. Accuracy

Accuracy answers:

> **Out of all predictions, how many were correct?**

Formula:

$$
Accuracy =
\frac{TP+TN}
{TP+TN+FP+FN}
$$

Using your example:

$$
Accuracy =
\frac{1+3}
{1+3+0+0}
=1
$$

So:

```text
Accuracy = 1.0 = 100%
```

### Remember

```text
Accuracy
    ↓
Overall correctness
```

### Problem with Accuracy

If your dataset is highly imbalanced, accuracy can be misleading.

Example:

```text
100 samples
95 = Class 0
5  = Class 1
```

A model that predicts **everything as Class 0** gets:

```text
95 / 100 = 95% accuracy
```

But it completely misses Class 1.

That's why we also use Precision, Recall and F1.

---

# 3. Precision

Precision answers:

> **Of everything the model predicted as Class 1, how many were actually Class 1?**

Formula:

$$
Precision =
\frac{TP}{TP+FP}
$$

Your example:

$$
Precision =
\frac{1}{1+0}
=1
$$

So:

```text
Precision = 1.0
```

### What matters?

Precision focuses on:

```text
TP vs FP
```

```text
Predicted Positive
       ↓
   ┌───┴───┐
   TP      FP
 Correct   Wrong
```

If **FP increases**, Precision decreases.

### Easy memory

**Precision → "When I predicted positive, was I right?"**

---

# 4. Recall

Recall answers:

> **Of all the actual Class 1 samples, how many did the model successfully find?**

Formula:

$$
Recall =
\frac{TP}{TP+FN}
$$

Your example:

$$
Recall =
\frac{1}{1+0}
=1
$$

So:

```text
Recall = 1.0
```

### What matters?

Recall focuses on:

```text
TP vs FN
```

```text
Actual Positive
       ↓
   ┌───┴───┐
   TP      FN
 Found    Missed
```

If **FN increases**, Recall decreases.

### Easy memory

**Recall → "Of all actual positives, how many did I find?"**

---

# 5. Precision vs Recall

This is very important for interviews.

### Precision

```text
Look at predicted positives
        ↓
TP / (TP + FP)
```

Question:

> "When I said positive, how often was I correct?"

### Recall

```text
Look at actual positives
        ↓
TP / (TP + FN)
```

Question:

> "Of all actual positives, how many did I find?"

So:

```text
Precision → FP is important
Recall    → FN is important
```

---

# 6. F1-Score

F1-score combines **Precision and Recall**.

Formula:

$$
F1 =
2\times
\frac{Precision\times Recall}
{Precision+Recall}
$$

Your values:

```text
Precision = 1
Recall    = 1
```

Therefore:

$$
F1 =
2\times\frac{1\times1}{1+1}
=1
$$

So:

```text
F1 = 1.0
```

### Why not simply average Precision and Recall?

Because F1 uses the **harmonic mean**, which penalizes a large imbalance between Precision and Recall.

For example:

```text
Precision = 1.0
Recall    = 0.1
```

F1 will be low.

So:

```text
High Precision + High Recall
            ↓
        High F1
```

### Easy memory

```text
F1
 ↓
Balance between Precision and Recall
```

---

# 7. ROC-AUC

ROC-AUC is different from the previous four.

For:

```text
Accuracy
Precision
Recall
F1
```

we normally use the **final predicted classes**:

```text
0 or 1
```

For ROC-AUC, we use the model's **probability scores**.

Example:

```text
Actual:       0    0    1    0

Probability:
             0.01 0.03 0.74 0.28
```

These probabilities come from:

```python
y_prob = model.predict_proba(X_test)[:, 1]
```

---

## What does ROC-AUC tell us?

It tells us how well the model can **separate Class 0 from Class 1 across different thresholds**.

Remember that we used:

```text
threshold = 0.5
```

But we could also use:

```text
0.3
0.4
0.5
0.6
0.7
...
```

Changing the threshold changes which samples become Class 1.

ROC-AUC evaluates the model's separation ability across these possible thresholds.

### ROC-AUC interpretation

Generally:

```text
ROC-AUC = 1.0
→ Excellent separation

ROC-AUC ≈ 0.9
→ Very good

ROC-AUC ≈ 0.8
→ Good

ROC-AUC ≈ 0.7
→ Moderate

ROC-AUC ≈ 0.5
→ Random-like

ROC-AUC < 0.5
→ Worse than random / ranking is reversed
```

---

# 8. Why ROC-AUC uses probabilities

Suppose we have:

```text
Actual:          1      0
Probability:   0.90   0.20
```

The model gives the actual positive a high probability and the negative a low probability.

That's good separation.

But imagine:

```text
Actual:          1      0
Probability:   0.55   0.45
```

Still correctly separated, but only slightly.

ROC-AUC is useful because it considers the **ranking/separation of the scores**, rather than only one chosen threshold.

---

# 9. All metrics together

Your complete classification evaluation is:

```text
                    Model
                      ↓
                 model.fit()
                      ↓
                  Prediction
                      ↓
            ┌─────────┴─────────┐
            ↓                   ↓
      model.predict()    predict_proba()
            ↓                   ↓
       0 / 1 classes       probabilities
            ↓                   ↓
     Confusion Matrix        ROC-AUC
            ↓
       TN FP FN TP
            ↓
    ┌───────┼────────┬────────┐
    ↓       ↓        ↓        ↓
Accuracy Precision Recall    F1
```

---

# 10. Formulas — Quick Revision

| Metric        | Formula                                             | Main focus                         |
| ------------- | --------------------------------------------------- | ---------------------------------- |
| **Accuracy**  | \(\frac{TP+TN}{TP+TN+FP+FN}\)                       | Overall correctness                |
| **Precision** | \(\frac{TP}{TP+FP}\)                                | Avoiding FP                        |
| **Recall**    | \(\frac{TP}{TP+FN}\)                                | Avoiding FN                        |
| **F1**        | \(2\frac{Precision\cdot Recall}{Precision+Recall}\) | Precision + Recall balance         |
| **ROC-AUC**   | Area under ROC curve                                | Class separation across thresholds |

And the most important thing to remember:

```text
Confusion Matrix
       ↓
TN, FP, FN, TP
       ↓
Accuracy
Precision
Recall
F1
```

while:

```text
Probability scores
       ↓
ROC curve
       ↓
ROC-AUC
```

This is the complete metric foundation you need for the classification models we're covering.

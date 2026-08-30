import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# 1. Load dataset
df = pd.read_csv(
    "../../data_classification/simple_classification_students.csv"
)


# 2. Features and target
X = df[["Study_Hours"]]
y = df["Result"]


# 3. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 4. Feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 5. Create SVC model
model = SVC(
    kernel="linear",
    C=1.0
)


# 6. Train
model.fit(X_train_scaled, y_train)


# 7. Prediction
y_pred = model.predict(X_test_scaled)

print("predicted class:", y_pred)
print("actual class:", y_test.to_numpy())


# 8. Metrics
print("---------------------------metrics---------------------------")

confusion_metrics = confusion_matrix(y_test, y_pred)
print("confusion matrix:", confusion_metrics)

accuracy = accuracy_score(y_test, y_pred)
print("accuracy:", accuracy)

precision = precision_score(y_test, y_pred)
print("precision:", precision)

recall = recall_score(y_test, y_pred)
print("recall:", recall)

f1 = f1_score(y_test, y_pred)
print("f1:", f1)


# 9. Decision score
y_score = model.decision_function(X_test_scaled)

print("decision score:", y_score)


# 10. ROC-AUC
roc_auc = roc_auc_score(y_test, y_score)

print("roc_auc:", roc_auc)



'''SVC
 ↓
model.fit()
 ↓
Decision Boundary + Maximum Margin
 ↓
model.predict()
 ↓
Class
 ↓
decision_function()
 ↓
Decision Score
 ↓
ROC-AUC'''
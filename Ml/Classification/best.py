import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# =========================================================
# CHANGE THIS: "simple" or "multiple"
# =========================================================

DATASET = "multiple"


if DATASET == "simple":

    df = pd.read_csv("../data_classification/simple_classification_students.csv")

    X = df[["Study_Hours"]]
    y = df["Result"]

else:

    df = pd.read_csv("../data_classification/multiple_classification_students.csv")

    X = df[
        [
            "Study_Hours",
            "Previous_Marks",
            "Attendance_Percent",
        ]
    ]

    y = df["Result"]


# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)


# =========================================================
# MODELS
# =========================================================

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression()),
    ]),

    "Decision Tree": DecisionTreeClassifier(
        criterion="gini",
        random_state=42,
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=550,
        random_state=42,
    ),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(
            n_neighbors=3,
            metric="euclidean",
        )),
    ]),

    "SVC": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            kernel="rbf",
            C=1.0,
            probability=True,
            random_state=42,
        )),
    ]),
}


# =========================================================
# MODEL COMPARISON
# =========================================================

results = []


for name, model in models.items():

    model.fit(X_train, y_train)

    # Predicted classes
    y_pred = model.predict(X_test)

    # Probability of Class 1
    y_proba = model.predict_proba(X_test)[:, 1]

    results.append({

        "Model": name,

        "Accuracy": accuracy_score(
            y_test,
            y_pred
        ),

        "Precision": precision_score(
            y_test,
            y_pred
        ),

        "Recall": recall_score(
            y_test,
            y_pred
        ),

        "F1": f1_score(
            y_test,
            y_pred
        ),

        "ROC-AUC": roc_auc_score(
            y_test,
            y_proba
        ),
    })


# =========================================================
# RESULTS TABLE
# =========================================================

results_df = pd.DataFrame(results)

print(f"\n=== Classification Comparison on {DATASET} dataset ===")

print(
    results_df.to_string(index=False)
)


# =========================================================
# BEST MODEL FOR EACH METRIC
# =========================================================

print(
    "\nBest Accuracy :",
    results_df.loc[
        results_df["Accuracy"].idxmax(),
        "Model"
    ]
)

print(
    "Best Precision:",
    results_df.loc[
        results_df["Precision"].idxmax(),
        "Model"
    ]
)

print(
    "Best Recall   :",
    results_df.loc[
        results_df["Recall"].idxmax(),
        "Model"
    ]
)

print(
    "Best F1       :",
    results_df.loc[
        results_df["F1"].idxmax(),
        "Model"
    ]
)

print(
    "Best ROC-AUC  :",
    results_df.loc[
        results_df["ROC-AUC"].idxmax(),
        "Model"
    ]
)
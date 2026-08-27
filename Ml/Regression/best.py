import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
)

# =========================================================
# CHANGE THIS: "simple" or "multiple"
# =========================================================
DATASET = "multiple"   # "simple" or "multiple"

if DATASET == "simple":
    df = pd.read_csv("../data/simple_linear_regression_students.csv")
    X = df[["Study_Hours"]]
    y = df["Marks"]
else:
    df = pd.read_csv("../data/multiple_linear_regression_students.csv")
    X = df[["Study_Hours", "Previous_Marks", "Attendance_Percent"]]
    y = df["Final_Marks"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Same settings as your project scripts
# KNN + SVR use scaling (Pipeline) so comparison is fair
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=550, random_state=42),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsRegressor(n_neighbors=2, metric="euclidean")),
    ]),
    "SVR": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVR(kernel="linear", C=1.0, epsilon=0.1)),
    ]),
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": root_mean_squared_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred),
    })

results_df = pd.DataFrame(results)
print(f"\n=== Comparison on {DATASET} dataset ===")
print(results_df.to_string(index=False))

print("\nBest MAE :", results_df.loc[results_df["MAE"].idxmin(), "Model"])
print("Best MSE :", results_df.loc[results_df["MSE"].idxmin(), "Model"])
print("Best RMSE:", results_df.loc[results_df["RMSE"].idxmin(), "Model"])
print("Best R2  :", results_df.loc[results_df["R2"].idxmax(), "Model"])

# Main winner for your project: lowest RMSE (check R2 too)
best = results_df.loc[results_df["RMSE"].idxmin()]
print("\n>>> Overall recommended (lowest RMSE):", best["Model"])
print(best)
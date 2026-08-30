A hyperparameter is a model setting chosen by the developer that controls the learning process or model behavior. Unlike model parameters, hyperparameters are not learned directly from the training data

this is a hyperparameter

Hyperparameter
→ chosen by us
→ controls the learning process

Parameter
→ learned by the model from training data

scalling :

We scale features so that their different numerical ranges don't unfairly dominate distance-based calculations



---------------------------------------------


| Model                   | Important Hyperparameters                                         | Output for threshold                    | Metrics                                  |
| ----------------------- | ----------------------------------------------------------------- | --------------------------------------- | ---------------------------------------- |
| **Logistic Regression** | `C`, `penalty`, `solver`, `max_iter`                              | Probability                             | Accuracy, Precision, Recall, F1, ROC-AUC |
| **KNN**                 | `n_neighbors`, `weights`, `metric`                                | Probability                             | Same                                     |
| **Decision Tree**       | `max_depth`, `min_samples_split`, `min_samples_leaf`, `criterion` | Probability                             | Same                                     |
| **Random Forest**       | `n_estimators`, `max_depth`, `min_samples_split`, `max_features`  | Probability                             | Same                                     |
| **SVM**                 | `C`, `kernel`, `gamma`                                            | Decision score / probability if enabled | Same                                     |




Hyperparameters
→ control how the model is built/tuned

Threshold
→ converts probability/score into Class 0 or 1

Metrics
→ measure how good the final predictions are


Pandas   → Data
NumPy    → Numbers / Arrays / Mathematics
Matplotlib → Graphs / Visualization
sklearn  → ML Algorithms



Regression
MAE   → LOWER ✅
MSE   → LOWER ✅
RMSE  → LOWER ✅
R²    → HIGHER ✅


Accuracy   → HIGHER ✅
Precision  → HIGHER ✅
Recall     → HIGHER ✅
F1         → HIGHER ✅
ROC-AUC    → HIGHER ✅
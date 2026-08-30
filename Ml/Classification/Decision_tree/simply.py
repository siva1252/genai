import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
#from sklearn.tree import export_text  #(for gini tree rules this one)
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,roc_auc_score





df = pd.read_csv("../../data_classification/simple_classification_students.csv")

x=df[['Study_Hours']]
y=df['Result']



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# print("total rows: ",len(x))
# print("total rows in train: ",len(x_train))
# print("total rows in test: ",len(x_test))

model = DecisionTreeClassifier(criterion='gini', random_state=42)

# print("model: ",model)

model.fit(x_train, y_train)


# tree_rules = export_text(model, feature_names=["Study_Hours"])

# print("tree_rules: ",tree_rules)

# tree = model.tree_

# print("Feature:", tree.feature)
# print("Threshold:", tree.threshold)
# print("Gini:", tree.impurity)
# print("Samples:", tree.n_node_samples)
# print("Classes:", tree.value)

#this one is how work in inside model.fit() for this thing to do gini thing


y_pred = model.predict(x_test)

# print("predicted class: ",y_pred)
# print("actual class: ",y_test.to_numpy())


print("________________metrics________________")

confusion_metrics = confusion_matrix(y_test,y_pred)
print("confusion matrix: ",confusion_metrics)

accuracy = accuracy_score(y_test,y_pred)
print("accuracy: ",accuracy)

precision = precision_score(y_test,y_pred)
print("precision: ",precision)

recall = recall_score(y_test,y_pred)
print("recall: ",recall)

f1 = f1_score(y_test,y_pred)
print("f1: ",f1)



#if we need to ROC-AUC score we can change your model.predict() to model.predict_proba() beacyse this things need probability not class

y_proba = model.predict_proba(x_test)[:,1]
print("probability: ",y_proba)

roc_auc = roc_auc_score(y_test,y_proba)
print("roc_auc: ",roc_auc)


print("________________total rows________________")

print("total rows: ",len(df))
print("total rows in train: ",len(x_train))
print("total rows in test: ",len(x_test))   



'''Dataset
   ↓
X = Study_Hours
y = Result
   ↓
Train/Test Split
   ↓
DecisionTreeClassifier
   ↓
model.fit(X_train, y_train)
   ↓
Tree learns internally
   │
   ├── Class proportions
   ├── Gini impurity
   ├── Feature
   ├── Threshold
   ├── Best split
   └── Repeat recursively
   ↓
model.predict(X_test)
   ↓
Predicted classes
   ↓
Compare with y_test
   ↓
┌─────────────────────────────┐
│ Confusion Matrix             │
│ Accuracy                     │
│ Precision                    │
│ Recall                       │
│ F1-score                     │
└─────────────────────────────┘
   ↓
model.predict_proba(X_test)
   ↓
Probability of class 1
   ↓
ROC-AUC'''
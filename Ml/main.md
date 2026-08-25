Linear Regression
→ coefficients + intercept

Decision Tree
→ thresholds + impurity + gain

Random Forest
→ multiple Decision Trees + averaging

Knn model 
-- finding the `K` nearest training samples and averaging their target values

#in code we write this things in from sklearn
linear_model
→ Linear-based algorithms

tree
→ Tree-based algorithms

ensemble
→ Multiple models combined together



#regressions of descion and random 
max_depth controls how deep the tree can grow.
min_samples_split checks whether the current node has enough samples to split.
min_samples_leaf checks whether both children created by a split have enough samples. If not, another threshold can be considered.
max_features randomly selects how many features are considered at each split.
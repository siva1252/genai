LINEAR
model.fit()
→ optimize w,b using regression loss
→ predict continuous value

LOGISTIC
model.fit()
→ optimize w,b using Log Loss
→ sigmoid gives probability
→ threshold gives class



                         DECISION TREE
                              ↓
                    Feature + Threshold
                              ↓
                         Split Data
                              ↓
                    ┌─────────┴─────────┐
                    ↓                   ↓
                REGRESSION         CLASSIFICATION
                    ↓                   ↓
              Numeric Target        Class Target
                    ↓                   ↓
            Target Mean          Class Proportions
                    ↓                   ↓
         Squared Error / MSE     Gini / Entropy
                    ↓                   ↓
          Weighted Child Error   Weighted Child Impurity
                    ↓                   ↓
              Best Split          Best Split
                    ↓                   ↓
          Repeat Recursively     Repeat Recursively
                    ↓                   ↓
               Final Leaf          Final Leaf
                    ↓                   ↓
          Mean → Prediction    Majority Class → Prediction
                    ↓                   ↓
                Number               Class



 The structure of Decision Tree Regression and Classification is similar: both recursively find feature-threshold splits. The main difference is that regression evaluates splits using numeric target variation such as squared error and predicts a numeric value at the leaf, whereas classification evaluates splits using class impurity such as Gini or entropy and predicts a class at the leaf.”


 REGRESSION
→ Number
→ Mean
→ MSE / Squared Error
→ Leaf = Number

CLASSIFICATION
→ Class
→ Class Proportions
→ Gini / Entropy
→ Leaf = Class          





Random Forest
      ↓
Many Decision Trees
      ↓
┌───────────────────┬───────────────────┐
↓                   ↓
Regression          Classification
↓                   ↓
Each tree gives     Each tree gives
a number            a class
↓                   ↓
Average all         Majority voting
tree outputs        ↓
↓                   Final class
Final number


Random Forest Regression → Average of tree predictions.
Random Forest Classification → Majority vote of tree predictions.

Decision Tree
→ one tree → one class

Random Forest
→ many trees → majority vote → final class

Decision Tree
→ one tree → one number

Random Forest
→ many trees → average → final number


                    KNN
                     ↓
             Find K nearest points
                     ↓
             Check their targets
                     ↓
          ┌──────────┴──────────┐
          ↓                     ↓
      Regression          Classification
          ↓                     ↓
    Target = number        Target = class
          ↓                     ↓
       Average             Majority Vote
          ↓                     ↓
    Number output          Class output


                             KNN
                          ↓
                    Input Features
                          ↓
                   Scale if needed
                          ↓
                     New sample
                          ↓
                Calculate distances
                          ↓
                Find K nearest points
                          ↓
                 Check target values
                          ↓
             ┌────────────┴────────────┐
             ↓                         ↓
        Regression                Classification
             ↓                         ↓
      Target = numeric            Target = class
             ↓                         ↓
          Average                Majority Vote
             ↓                         ↓
       Number prediction         Class prediction


       Logistic Regression
→ probability-oriented classification

SVC
→ maximum-margin classification


                    SVM
                     ↓
          ┌──────────┴──────────┐
          ↓                     ↓
         SVC                   SVR
   Classification            Regression
          ↓                     ↓
    Class target             Numeric target
          ↓                     ↓
   Decision boundary        Regression function
          ↓                     ↓
   Maximum margin           ε-insensitive tube
          ↓                     ↓
   Class 0 / Class 1        Numeric value


   SVC
→ "Which class does this point belong to?"
→ Decision boundary
→ Maximum margin
→ Class output


SVR
→ "What numeric value should I predict?"
→ Regression function
→ ε-insensitive margin/tube
→ Numeric output



|                   | SVC                          | SVR                                      |
| ----------------- | ---------------------------- | ---------------------------------------- |
| Purpose           | Classification               | Regression                               |
| Target            | Class                        | Number                                   |
| Main idea         | Maximum-margin boundary      | ε-insensitive regression                 |
| Output            | Class                        | Numeric value                            |
| Example           | Pass / Fail                  | Salary / Price                           |
| Important concept | Support vectors + margin     | Support vectors + ε-tube                 |
| `C`               | Error vs margin trade-off    | Error outside ε-tube vs model complexity |
| Kernel            | Can use linear/RBF/poly etc. | Can use linear/RBF/poly etc.             |




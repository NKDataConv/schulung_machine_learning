import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score
from datenmanagement_wetter import (x_train, x_test, x_vali,
                                   y_train, y_test, y_vali)

PARAMS = {"max_depth": 4}

# classifier = DecisionTreeClassifier(**PARAMS)
classifier = SVC()

classifier.fit(x_train, y_train)

y_pred_train = classifier.predict(x_train)
df_predictions = pd.DataFrame({"actual": y_train,
                               "prediction": y_pred_train})
cm = pd.crosstab(index=df_predictions["prediction"],
                    columns=df_predictions["actual"],
                    margins=True)
print(cm)

precision_train = precision_score(df_predictions["actual"], df_predictions["prediction"], pos_label=0)
recall_train = recall_score(df_predictions["actual"], df_predictions["prediction"], pos_label=0)
accuray_train = accuracy_score(df_predictions["actual"], df_predictions["prediction"])
print("Training Precision: ", precision_train)
print("Training Recall: ", recall_train)
print("Training Accuracy: ", accuray_train)

### Validation Performance
y_pred_vali = classifier.predict(x_vali)

df_predictions_vali = pd.DataFrame({"actual": y_vali,
                                    "prediction": y_pred_vali})
cm_vali = pd.crosstab(index=df_predictions_vali["prediction"],
                    columns=df_predictions_vali["actual"],
                    margins=True)
print(cm_vali)
precision_vali = precision_score(df_predictions_vali["actual"], df_predictions_vali["prediction"], pos_label=0)
recall_vali = recall_score(df_predictions_vali["actual"], df_predictions_vali["prediction"], pos_label=0)
accuray_vali = accuracy_score(df_predictions_vali["actual"], df_predictions_vali["prediction"])
print("Validierung Precision: ", precision_vali)
print("Validierung Recall: ", recall_vali)
print("Validierung Accuracy: ", accuray_vali)

print("Overfitting accuracy: ", accuray_train - accuray_vali)

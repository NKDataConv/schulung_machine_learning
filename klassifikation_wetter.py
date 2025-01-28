import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from datenmanagement_wetter import (x_train, x_test, x_vali,
                                   y_train, y_test, y_vali)
from sklearn.model_selection import cross_validate

PARAMS = {"max_depth": 4}

# classifier = DecisionTreeClassifier(**PARAMS)
classifier = SVC()

# classifier_cv = cross_validate(classifier, x_train, y_train, cv=5, scoring="recall")
# overall_recall = classifier_cv["test_score"].mean()
# print("Recall: ", overall_recall)
# print(classifier_cv["test_score"])



# alternativ:
# classifier = DecisionTreeClassifier(max_depth=4)

classifier.fit(x_train, y_train)

# CUTOFF = 0.2
# y_pred_train_proba = classifier.predict_proba(x_train)
# y_pred_train = y_pred_train_proba[:, 1] >= CUTOFF

y_pred_train = classifier.predict(x_train)

correct_classified = y_pred_train == y_train
# print(correct_classified.sum())
# print(len(x_train))
df_predictions = pd.DataFrame({"actual": y_train,
                               "prediction": y_pred_train})

cm = pd.crosstab(index=df_predictions["prediction"],
                    columns=df_predictions["actual"],
                    margins=True)
print(cm)

from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score
precision_train = precision_score(df_predictions["actual"], df_predictions["prediction"], pos_label=0)
recall_train = recall_score(df_predictions["actual"], df_predictions["prediction"], pos_label=0)
accuray_train = accuracy_score(df_predictions["actual"], df_predictions["prediction"])
auc_score_train = roc_auc_score(df_predictions["actual"], y_pred_train_proba[:, 1])
print("Training Precision: ", precision_train)
print("Training Recall: ", recall_train)
print("Training Accuracy: ", accuray_train)
print("Training Area under the curve: ", auc_score_train)

### Validation Performance
# y_pred_vali = classifier.predict_proba(x_vali)
# print(y_pred_vali)
# y_pred_vali = y_pred_vali[:, 1] >= CUTOFF
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


# precision_score = 25318 / 29934
# print(precision_score)
# precision_score = cm[0][0] / cm["All"][0]
# print(precision_score)

# accuracy_train = correct_classified.sum() / len(x_train)
# print("Auf den Trainingsdaten hat er eine Accuracy von ", accuracy_train)
# # print(len(y_pred_train))

# y_pred_test = classifier.predict(x_test)
# correct_classified = y_pred_test == y_test
# print(correct_classified.sum())
# print(len(correct_classified))

# accuracy_test = correct_classified.sum() / len(correct_classified)
# print("Der Algorithmus funktioniert mit einer Accuracy von: ", accuracy_test)
#
# print("Overfitting: ", accuracy_train - accuracy_test)

# from sklearn.tree import plot_tree
# import matplotlib.pyplot as plt
# plot_tree(classifier)
# plt.show()

# classifier.feature_importances_
#
# df_feature_importance = pd.DataFrame({"Feature": x_train.columns,
#                                         "Importance": classifier.feature_importances_})
# print(df_feature_importance)
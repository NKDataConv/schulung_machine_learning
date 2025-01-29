import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score
from datenmanagement_wetter import (x_train, x_test, x_vali,
                                   y_train, y_test, y_vali)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, accuracy_score

# param_grid = {"max_depth": [4, 5, 6],
#               "n_estimators": [50, 100, 150],
#               "min_samples_split": [2, 3]}

scorer = make_scorer(accuracy_score)

classifier = MLPClassifier(hidden_layer_sizes=(200, 100, 100, 50, 10), max_iter=2000)
# random_search = RandomizedSearchCV(classifier, scoring=scorer, cv=3, param_distributions=param_grid, n_iter=4)
classifier.fit(x_train, y_train)

# print(random_search.best_params_)
# classifier = random_search.best_estimator_


mlflow.start_run()

# PARAMS = {"max_depth": 3,
#           "n_estimators": 100}

# for key, value in random_search.best_params_.items():
#     mlflow.log_param(key, value)

# mlflow.log_param("n_estimators", PARAMS["n_estimators"])
# mlflow.log_param("max_depth", PARAMS["max_depth"])

# classifier = DecisionTreeClassifier(**PARAMS)
# classifier = SGDClassifier(**PARAMS)
# classifier = GradientBoostingClassifier(**PARAMS)

# classifier.fit(x_train, y_train)

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

mlflow.log_metric("accuray_vali", accuray_vali)

mlflow.end_run()

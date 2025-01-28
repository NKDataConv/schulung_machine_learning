# ----- CLASSIFICATION TREES
import mlflow
from sklearn.model_selection import train_test_split
from sklearn import tree, metrics
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from soccer_datenmanagement_cls import x, y, quotas, dat
from _utils.multiclass_prediction_from_probabilities import multiclass_prediction_from_probabilities
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score

mlflow.start_run()
# --- Splitting for getting training and test datasets. Also quotas object is splitted, which is required
# for model-evaluation later on.
x_train, x_test, y_train, y_test, quotas_train, quotas_test = train_test_split(x,
                                                    y,
                                                    quotas,
                                                    test_size=0.4,
                                                    random_state=9)

PARAMS = {"n_estimators": 1100,
          "max_depth": 8,
          "min_impurity_decrease": 0.0001,
          "min_samples_split":200}

# --- Make classification
cls = RandomForestClassifier(**PARAMS)
# cls = GradientBoostingClassifier()

for key, value in PARAMS.items():
    mlflow.log_param(key, value)

# --- Make classification
# cls = RandomForestClassifier(**params)
cls.fit(X=x_train, y=y_train)

# --- Make prediction
y_pred_train = cls.predict(X=x_train)
precision_train = precision_score(y_train, y_pred_train, pos_label=0, average="macro")
recall_train = recall_score(y_train, y_pred_train, pos_label=0, average="macro")
accuray_train = accuracy_score(y_train, y_pred_train)
print("Training Precision: ", precision_train)
print("Training Recall: ", recall_train)
print("Training Accuracy: ", accuray_train)

# --- Make Plot (can be removed when different models are build
# tree.plot_tree(cls)
# plt.show()

# --- Make prediction
probabilities = cls.predict_proba(x_test)

probabilities = pd.DataFrame({'1': probabilities[:,1],
                              '0': probabilities[:,0],
                              '2': probabilities[:,2]})

probabilities.set_index(x_test.index, inplace=True)


# Cutoffs - Here the minimum of probability can be set for each category.
# Example: The model must assign at least a probability of xyz in order to make a prediction.
cutoffs = {'1': 0.8, '0': 0.3, '2': 0.3}

# Create prediction of result-tendency of the game
probabilities['predicted'] = multiclass_prediction_from_probabilities(
                                                probabilities=probabilities,
                                                cutoffs=cutoffs)

probabilities['predicted'].value_counts() # Get info of how many games are predicted
probabilities['actual'] = y_test # Attach actual values to table
probabilities = pd.concat((probabilities, quotas_test), axis=1) # Attach quotas to table

probabilities.dropna(inplace=True) # Drop games that are not predicted

precision_test = precision_score(probabilities['actual'], probabilities['predicted'], pos_label=0, average="macro")
recall_test = recall_score(probabilities['actual'], probabilities['predicted'], pos_label=0, average="macro")
accuray_test = accuracy_score(probabilities['actual'], probabilities['predicted'])
print("Test Precision: ", precision_test)
print("Test Recall: ", recall_test)
print("Test Accuracy: ", accuray_test)
print("Overfitting: ", accuray_train - accuray_test)

# --- Generate confusion matrix
cm = pd.crosstab(index=probabilities['actual'], columns=probabilities['predicted'], margins = True)
cm = cm.rename(columns={0.0:'D',1.0:'H',2.0:'A'})
cm = cm.rename(index={0:'D',1:'H',2:'A'})
existent_cols = [i for i in ('H','D','A','All') if i in cm.columns]
cm.loc[existent_cols,existent_cols]
print(cm)

report = metrics.classification_report(y_true=probabilities['actual'],
                                       y_pred=probabilities['predicted'],
                                       zero_division=0)
print(report)

# --- Join Teamnames onto dataset
dat2join = dat.loc[:,['HeimTeam','AuswTeam']]
probabilities = probabilities.join(other=dat2join, how="left")

# --- Evaluation on quotas
# Attach information: What was the actual quota that won
probabilities['correct_bet'] = np.where(probabilities['predicted'] == probabilities['actual'], True, False)
probabilities['quota_in_game'] = np.nan
probabilities.loc[probabilities['actual'] == 1,'quota_in_game'] = probabilities['BW_H']
probabilities.loc[probabilities['actual'] == 0,'quota_in_game'] = probabilities['BW_D']
probabilities.loc[probabilities['actual'] == 2,'quota_in_game'] = probabilities['BW_A']

# Attach information: What was the predicted quota
probabilities['quota_predicted'] = np.nan
probabilities.loc[probabilities['predicted'] == 1,'quota_predicted'] = probabilities['BW_H']
probabilities.loc[probabilities['predicted'] == 0,'quota_predicted'] = probabilities['BW_D']
probabilities.loc[probabilities['predicted'] == 2,'quota_predicted'] = probabilities['BW_A']

probabilities['win'] = np.where(probabilities['correct_bet'],
         probabilities['quota_in_game'] * 100 - 100,
         -100)

# --- Amount of games
print("Games played: " + str(len(probabilities)))
print("Percent correct bets: " + str(probabilities['correct_bet'].mean().round(2)))
print("Average quota in game: " + str(probabilities['quota_in_game'].mean().round(2)))
print("Average quota predicted: " + str(probabilities['quota_predicted'].mean().round(2)))
print("Win/Loss: " + str(probabilities['win'].sum().round(2)))

mlflow.log_metric("Win / Loss", probabilities['win'].sum())

mlflow.end_run()

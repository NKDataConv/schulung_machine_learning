import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from datenmangement_wetter import (x_train, x_test, x_vali,
                                   y_train, y_test, y_vali)

PARAMS = {"max_depth": 4}

classifier = DecisionTreeClassifier(**PARAMS)
# alternativ:
# classifier = DecisionTreeClassifier(max_depth=4)

classifier.fit(x_train, y_train)

y_pred_train = classifier.predict(x_train)
# print(y_pred_train)
correct_classified = y_pred_train == y_train
# print(correct_classified.sum())
# print(len(x_train))
accuracy_train = correct_classified.sum() / len(x_train)
print("Auf den Trainingsdaten hat er eine Accuracy von ", accuracy_train)
# print(len(y_pred_train))

y_pred_test = classifier.predict(x_test)
correct_classified = y_pred_test == y_test
# print(correct_classified.sum())
# print(len(correct_classified))

accuracy_test = correct_classified.sum() / len(correct_classified)
print("Der Algorithmus funktioniert mit einer Accuracy von: ", accuracy_test)

print("Overfitting: ", accuracy_train - accuracy_test)

# from sklearn.tree import plot_tree
# import matplotlib.pyplot as plt
# plot_tree(classifier)
# plt.show()

classifier.feature_importances_

df_feature_importance = pd.DataFrame({"Feature": x_train.columns,
                                        "Importance": classifier.feature_importances_})
print(df_feature_importance)
from klassifikation_wetter import y_pred_train_proba, y_train

# --------------- ROC-Chart
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import numpy as np
import matplotlib.pyplot as plt

_auc_score = roc_auc_score(y_true = y_train,
                      y_score = y_pred_train_proba[:, 1])

fig, axes = plt.subplots()

fpr, tpr, thresholds = roc_curve(y_true = y_train,
                                 y_score = y_pred_train_proba[:, 1])

axes.plot(fpr, tpr, color='darkorange', label='ROC curve (area = {:5.2f})'.format(_auc_score)) # label is for the legend
axes.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
axes.set_xlabel('False Positive Rate')
axes.set_ylabel('True Positive Rate')
axes.set_title('Receiver operating characteristic')
axes.legend(loc="lower right")
axes.set_xticks(fpr)
axes.set_yticks(tpr)
thresholds_2nd = np.round(thresholds,2)
for index, item in enumerate(thresholds_2nd):
    axes.annotate(str(item), xy=(fpr[index], tpr[index]))
plt.savefig("roc_curve.png")
# plt.show()
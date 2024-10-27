from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

df = pd.read_csv('data_metrics.csv')
df.head()
thresh = 0.5
df['predicted_RF'] = (df.model_RF >= 0.5).astype('int')
df['predicted_LR'] = (df.model_LR >= 0.5).astype('int')
df.head()
confusion_matrix(df.actual_label.values, df.predicted_RF.values)
def find_TP(y_true, y_pred):
    # counts the number of true positives (y_true = 1, y_pred = 1)
    return sum((y_true == 1) & (y_pred == 1))
def find_FN(y_true, y_pred):
    # counts the number of false negatives (y_true = 1, y_pred = 0)
    return sum((y_true == 1) & (y_pred == 0))
def find_FP(y_true, y_pred):
    # counts the number of false positives (y_true = 0, y_pred = 1)
    return sum((y_true == 0) & (y_pred == 1))
def find_TN(y_true, y_pred):
    # counts the number of true negatives (y_true = 0, y_pred = 0)
    return sum((y_true == 0) & (y_pred == 0))
print('TP:', find_TP(df.actual_label.values, df.predicted_RF.values))
print('FN:', find_FN(df.actual_label.values, df.predicted_RF.values))
print('FP:', find_FP(df.actual_label.values, df.predicted_RF.values))
print('TN:', find_TN(df.actual_label.values, df.predicted_RF.values))
def find_conf_matrix_values(y_true, y_pred):
    # calculate TP, FN, FP, TN
    TP = find_TP(y_true, y_pred)
    FN = find_FN(y_true, y_pred)
    FP = find_FP(y_true, y_pred)
    TN = find_TN(y_true, y_pred)
    return TP, FN, FP, TN

def trynov_confusion_matrix(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return np.array([[TN, FP], [FN, TP]])
trynov_confusion_matrix(df.actual_label.values, df.predicted_RF.values)
assert np.array_equal(trynov_confusion_matrix(df.actual_label.values, df.predicted_RF.values),
                       confusion_matrix(df.actual_label.values, df.predicted_RF.values)), \
                       'trynov_confusion_matrix() is not correct for RF'

assert np.array_equal(trynov_confusion_matrix(df.actual_label.values, df.predicted_LR.values),
                       confusion_matrix(df.actual_label.values, df.predicted_LR.values)), \
                       'trynov_confusion_matrix() is not correct for LR'

def trynov_accuracy_score(y_true, y_pred):
    # calculates the fraction of samples
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    total = TP + FN + FP + TN
    return (TP + TN) / total if total > 0 else 0

# Assertions to validate the accuracy score function
assert trynov_accuracy_score(df.actual_label.values, df.predicted_RF.values) == \
    accuracy_score(df.actual_label.values, df.predicted_RF.values), \
    'trynov_accuracy_score failed on RF'

assert trynov_accuracy_score(df.actual_label.values, df.predicted_LR.values) == \
    accuracy_score(df.actual_label.values, df.predicted_LR.values), \
    'trynov_accuracy_score failed on LR'

# Printing accuracy scores
print('Accuracy RF: %.3f' % trynov_accuracy_score(df.actual_label.values, df.predicted_RF.values))
print('Accuracy LR: %.3f' % trynov_accuracy_score(df.actual_label.values, df.predicted_LR.values))

def trynov_recall_score(y_true, y_pred):
    # Calculates the fraction of positive samples predicted correctly
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return TP / (TP + FN) if (TP + FN) > 0 else 0

# Assertions to validate the recall score function
assert trynov_recall_score(df.actual_label.values, df.predicted_RF.values) == \
    recall_score(df.actual_label.values, df.predicted_RF.values), \
    'trynov_recall_score failed on RF'

assert trynov_recall_score(df.actual_label.values, df.predicted_LR.values) == \
    recall_score(df.actual_label.values, df.predicted_LR.values), \
    'trynov_recall_score failed on LR'

# Print recall scores
print('Recall RF: %.3f' % trynov_recall_score(df.actual_label.values, df.predicted_RF.values))
print('Recall LR: %.3f' % trynov_recall_score(df.actual_label.values, df.predicted_LR.values))

def trynov_precision_score(y_true, y_pred):
    # Calculates the fraction of predicted positive samples that are actually positive
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return TP / (TP + FP) if (TP + FP) > 0 else 0

# Assertions to validate the precision score function
assert trynov_precision_score(df.actual_label.values, df.predicted_RF.values) == \
    precision_score(df.actual_label.values, df.predicted_RF.values), \
    'trynov_precision_score failed on RF'

assert trynov_precision_score(df.actual_label.values, df.predicted_LR.values) == \
    precision_score(df.actual_label.values, df.predicted_LR.values), \
    'trynov_precision_score failed on LR'

# Print precision scores
print('Precision RF: %.3f' % trynov_precision_score(df.actual_label.values, df.predicted_RF.values))
print('Precision LR: %.3f' % trynov_precision_score(df.actual_label.values, df.predicted_LR.values))

def trynov_f1_score(y_true, y_pred):
    # Calculates the F1 score
    recall = trynov_recall_score(y_true, y_pred)
    precision = trynov_precision_score(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Assertions to validate the F1 score function
assert trynov_f1_score(df.actual_label.values, df.predicted_RF.values) == \
    f1_score(df.actual_label.values, df.predicted_RF.values), \
    'trynov_f1_score failed on RF'

"""assert trynov_f1_score(df.actual_label.values, df.predicted_LR.values) == \
    f1_score(df.actual_label.values, df.predicted_LR.values), \
    'trynov_f1_score failed on LR'"""

# Print F1 scores
print('F1 RF: %.3f' % trynov_f1_score(df.actual_label.values, df.predicted_RF.values))
print('F1 LR: %.3f' % trynov_f1_score(df.actual_label.values, df.predicted_LR.values))

print('Scores with threshold = 0.5')

print('Accuracy RF: %.3f' % (trynov_accuracy_score(df.actual_label.values, df.predicted_RF.values)))
print('Recall RF: %.3f' % (trynov_recall_score(df.actual_label.values, df.predicted_RF.values)))
print('Precision RF: %.3f' % (trynov_precision_score(df.actual_label.values, df.predicted_RF.values)))
print('F1 RF: %.3f' % (trynov_f1_score(df.actual_label.values, df.predicted_RF.values)))

print()  # Пустая строка для разделения выводов

print('Scores with threshold = 0.25')

print('Accuracy RF: %.3f' % (trynov_accuracy_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int'))))
print('Recall RF: %.3f' % (trynov_recall_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int'))))
print('Precision RF: %.3f' % (trynov_precision_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int'))))
print('F1 RF: %.3f' % (trynov_f1_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int'))))

fpr_RF, tpr_RF, thresholds_RF = roc_curve(df.actual_label.values, df.model_RF.values)
fpr_LR, tpr_LR, thresholds_LR = roc_curve(df.actual_label.values, df.model_LR.values)

auc_RF = roc_auc_score(df.actual_label.values, df.model_RF.values)
auc_LR = roc_auc_score(df.actual_label.values, df.model_LR.values)

print('AUC RF: %.3f' % auc_RF)
print('AUC LR: %.3f' % auc_LR)

plt.plot(fpr_RF, tpr_RF, 'r-', label=f'RF AUC: {auc_RF:.3f}')
plt.plot(fpr_LR, tpr_LR, 'b-', label=f'LR AUC: {auc_LR:.3f}')
plt.plot([0, 1], [0, 1], 'k-', label='random')
plt.plot([0, 0, 1, 1], [0, 1, 1, 1], 'g-', label='perfect')

plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
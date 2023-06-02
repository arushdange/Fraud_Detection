import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt

data = pd.read_csv('creditcard.csv')

fraud = data[data['Class'] == 1]
non_fraud = data[data['Class'] == 0].sample(len(fraud))

data_balanced = pd.concat([fraud, non_fraud])

X = data_balanced.drop('Class', axis=1)
y = data_balanced['Class']

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
y_pred = lof.fit_predict(X)

# Predict anomaly scores for the test set
y_score = lof.negative_outlier_factor_ * -1

precision, recall, thresholds = precision_recall_curve(y, y_score)
auc_score = auc(recall, precision)

plt.plot(recall, precision, color='b', label='LOF (AUC = %0.2f)' % auc_score)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='lower left')
plt.show()

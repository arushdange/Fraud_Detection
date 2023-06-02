import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt

data = pd.read_csv('creditcard.csv')

fraud = data[data['Class'] == 1]
non_fraud = data[data['Class'] == 0].sample(len(fraud))

data_balanced = pd.concat([fraud, non_fraud])

X = data_balanced.drop('Class', axis=1)
y = data_balanced['Class']

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)

y_pred = kmeans.predict(X)

# Invert cluster labels to get outlier scores
y_score = 1 - y_pred

precision, recall, thresholds = precision_recall_curve(y, y_score)
auc_score = auc(recall, precision)

plt.plot(recall, precision, color='b', label='K-means (AUC = %0.2f)' % auc_score)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='lower left')
plt.show()

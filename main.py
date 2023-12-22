import csv
import random

import numpy as np
import pandas as pd
import numpy
from numpy import interp
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from src.read_data import DataSet
from src.clustering import PerformClustering

# Section 1: Instantiate DataSet class
"""
Instantiating the DataSet class will create an object. This object will contain all the datasets:
    
    dataset_object = DataSet()
    
    - dataset_object.disease_name()                
    - dataset_object.disease_size()                
    - dataset_object.snoRNA_name()                 
    - dataset_object.snoRNA_size()                 
    - dataset_object.known_association()           
    - dataset_object.SnoRNA_similarity()           
    - dataset_object.disease_similarity()          
    - dataset_object.disease_semantic_similarity() 
    - dataset_object.adjacency_matrix()            
    - dataset_object.snoRNA_functional_similarity()

"""

DataSet = DataSet()

# Section 2: Plot elbow curve to find the optimal number of clusters
"""
This function will plot the elbow curve for the given dataset and the given maximum number of clusters.
Then we can use the elbow curve to find the optimal number of clusters

    PerformClustering = PerformClustering(max_number_of_clusters= 50)
    
    - PerformClustering.plot_elbow_curve(major)
    - PerformClustering.prepare_data_for_clustering(
            adjacency_matrix, 
            disease_semantic_similarity, 
            snoRNA_functional_similarity
      )
    

"""
PerformClustering = PerformClustering()

unknown, known, major = PerformClustering.prepare_data_for_clustering(
    DataSet.adjacency_matrix,
    DataSet.disease_semantic_similarity,
    DataSet.snoRNA_functional_similarity,
    DataSet.snoRNA_size,
    DataSet.disease_size
)

# Section 3: Perform K-Means Clustering
"""
    This function will perform K-Means Clustering on the given dataset and the given number of clusters.
    Then it will return the clustered dataset.
    
    From this section we can get the:
    - labels
"""
labels = PerformClustering.perform_clustering(major)

# Section 4: Divide the clustered dataset into two groups and populate the variables with Random Selection
"""
    This section will divide the clustered dataset into X and Y coordinate groups.
    Considering 20 clusters we will have 20 X and 20 Y groups.


"""
type1_x = []
type1_y = []
type2_x = []
type2_y = []
type3_x = []
type3_y = []
type4_x = []
type4_y = []
type5_x = []
type5_y = []
type6_x = []
type6_y = []
type7_x = []
type7_y = []
type8_x = []
type8_y = []
type9_x = []
type9_y = []
type10_x = []
type10_y = []
type11_x = []
type11_y = []
type12_x = []
type12_y = []
type13_x = []
type13_y = []
type14_x = []
type14_y = []
type15_x = []
type15_y = []
type16_x = []
type16_y = []
type17_x = []
type17_y = []
type18_x = []
type18_y = []
type19_x = []
type19_y = []
type20_x = []
type20_y = []

for i in range(len(labels)):
    if labels[i] == 0:
        type1_x.append(unknown[i][0])
        type1_y.append(unknown[i][1])
    if labels[i] == 1:
        type2_x.append(unknown[i][0])
        type2_y.append(unknown[i][1])
    if labels[i] == 2:
        type3_x.append(unknown[i][0])
        type3_y.append(unknown[i][1])
    if labels[i] == 3:
        type4_x.append(unknown[i][0])
        type4_y.append(unknown[i][1])
    if labels[i] == 4:
        type5_x.append(unknown[i][0])
        type5_y.append(unknown[i][1])
    if labels[i] == 5:
        type6_x.append(unknown[i][0])
        type6_y.append(unknown[i][1])
    if labels[i] == 6:
        type7_x.append(unknown[i][0])
        type7_y.append(unknown[i][1])
    if labels[i] == 7:
        type8_x.append(unknown[i][0])
        type8_y.append(unknown[i][1])
    if labels[i] == 8:
        type9_x.append(unknown[i][0])
        type9_y.append(unknown[i][1])
    if labels[i] == 9:
        type10_x.append(unknown[i][0])
        type10_y.append(unknown[i][1])
    if labels[i] == 10:
        type11_x.append(unknown[i][0])
        type11_y.append(unknown[i][1])
    if labels[i] == 11:
        type12_x.append(unknown[i][0])
        type12_y.append(unknown[i][1])
    if labels[i] == 12:
        type13_x.append(unknown[i][0])
        type13_y.append(unknown[i][1])
    if labels[i] == 13:
        type14_x.append(unknown[i][0])
        type14_y.append(unknown[i][1])
    if labels[i] == 14:
        type15_x.append(unknown[i][0])
        type15_y.append(unknown[i][1])
    if labels[i] == 15:
        type16_x.append(unknown[i][0])
        type16_y.append(unknown[i][1])
    if labels[i] == 16:
        type17_x.append(unknown[i][0])
        type17_y.append(unknown[i][1])
    if labels[i] == 17:
        type18_x.append(unknown[i][0])
        type18_y.append(unknown[i][1])
    if labels[i] == 18:
        type19_x.append(unknown[i][0])
        type19_y.append(unknown[i][1])
    if labels[i] == 19:
        type20_x.append(unknown[i][0])
        type20_y.append(unknown[i][1])

# Section 5:
types = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
dataset = []
randomly_selected_types = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]

for k1 in range(len(type1_x)):
    types[0].append((type1_x[k1], type1_y[k1]))
for k2 in range(len(type2_x)):
    types[1].append((type2_x[k2], type2_y[k2]))
for k3 in range(len(type3_x)):
    types[2].append((type3_x[k3], type3_y[k3]))
for k4 in range(len(type4_x)):
    types[3].append((type4_x[k4], type4_y[k4]))
for k5 in range(len(type5_x)):
    types[4].append((type5_x[k5], type5_y[k5]))
for k6 in range(len(type6_x)):
    types[5].append((type6_x[k6], type6_y[k6]))
for k7 in range(len(type7_x)):
    types[6].append((type7_x[k7], type7_y[k7]))
for k8 in range(len(type8_x)):
    types[7].append((type8_x[k8], type8_y[k8]))
for k9 in range(len(type9_x)):
    types[8].append((type9_x[k9], type9_y[k9]))
for k10 in range(len(type10_x)):
    types[9].append((type10_x[k10], type10_y[k10]))
for k11 in range(len(type11_x)):
    types[10].append((type11_x[k11], type11_y[k11]))
for k12 in range(len(type12_x)):
    types[11].append((type12_x[k12], type12_y[k12]))
for k13 in range(len(type13_x)):
    types[12].append((type13_x[k13], type13_y[k13]))
for k14 in range(len(type14_x)):
    types[13].append((type14_x[k14], type14_y[k14]))
for k15 in range(len(type15_x)):
    types[14].append((type15_x[k15], type15_y[k15]))
for k16 in range(len(type16_x)):
    types[15].append((type16_x[k16], type16_y[k16]))
for k17 in range(len(type17_x)):
    types[16].append((type17_x[k17], type17_y[k17]))
for k18 in range(len(type18_x)):
    types[17].append((type18_x[k18], type18_y[k18]))
for k19 in range(len(type19_x)):
    types[18].append((type19_x[k19], type19_y[k19]))
for k20 in range(len(type20_x)):
    types[19].append((type20_x[k20], type20_y[k20]))

for i in range(20):
    randomly_selected_types[i] = random.sample(types[i], int((len(types[i]) / len(labels)) * len(known)))

for i in range(DataSet.disease_size):
    for j in range(DataSet.snoRNA_size):
        for k in range(20):
            if (i, j) in randomly_selected_types[k]:
                dataset.append((i, j))  # Store the randomly extracted 23X240 samples in the dataset

for i in range(DataSet.snoRNA_size):
    for j in range(DataSet.disease_size):
        if DataSet.adjacency_matrix[i, j] == 1:
            dataset.append((i, j))  # Combine Major and Minor into training samples containing 10,950 samples

length = len(dataset)

sumy1 = numpy.zeros(length)
sumy2 = numpy.zeros(length)
sumy3 = numpy.zeros(length)
sumy4 = numpy.zeros(length)
sumy5 = numpy.zeros(length)
sumy6 = numpy.zeros(length)
sumy7 = numpy.zeros(length)
sumy8 = numpy.zeros(length)
sumy9 = numpy.zeros(length)
sumy10 = numpy.zeros(length)
sumy11 = numpy.zeros(length)
sumy12 = numpy.zeros(length)
sumy13 = numpy.zeros(length)
sumy14 = numpy.zeros(length)
sumy15 = numpy.zeros(length)
sumy16 = numpy.zeros(length)
sumy17 = numpy.zeros(length)
sumy18 = numpy.zeros(length)
sumy19 = numpy.zeros(length)
sumy20 = numpy.zeros(length)

x = []
x1 = []
x2 = []
y = []


for i in dataset:
    q = DataSet.disease_semantic_similarity[i[1], :].tolist() + DataSet.snoRNA_functional_similarity[i[0], :].tolist()
    x.append(q)
    if (i[0], i[1]) in known:
        y.append(1)
    else:
        y.append(0)



ys = numpy.array(y)
xs = numpy.array(x)







GBDT = GradientBoostingClassifier(n_estimators=12, max_depth=5, min_samples_leaf=13)
GBDT.fit(xs, ys)
OHE = OneHotEncoder()
OHE.fit(GBDT.apply(xs)[:, :, 0])
LR = LogisticRegression()
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

X_train, X_test, y_train, y_test = train_test_split(xs, ys, test_size=0.2, random_state=42)

LR.fit(OHE.transform(GBDT.apply(X_train)[:, :, 0]), y_train)
probas_ = LR.predict_proba(OHE.transform(GBDT.apply(X_test)[:, :, 0]))
fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
tprs.append(interp(mean_fpr, fpr, tpr))
tprs[-1][0] = 0.0
roc_auc = auc(fpr, tpr)
aucs.append(roc_auc)

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you have predictions and true labels
# Replace predictions and true labels with your actual values
predictions = LR.predict(OHE.transform(GBDT.apply(X_test)[:, :, 0]))
true_labels = y_test

# Calculate confusion matrix
conf_matrix = confusion_matrix(true_labels, predictions)

# Display confusion matrix using heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

# Assuming you have predictions and true labels
# Replace predictions and true labels with your actual values
predictions = LR.predict_proba(OHE.transform(GBDT.apply(X_test)[:, :, 0]))[:, 1]
true_labels = y_test

precision, recall, _ = precision_recall_curve(true_labels, predictions)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='b', label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp

# Assuming you have defined tprs, aucs, mean_fpr previously
mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)

plt.figure(figsize=(8, 6))
plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend()
plt.show()

gbm0 = GradientBoostingClassifier(
    n_estimators=12,
    max_depth=5,
    min_samples_leaf=13
)

gbm0.fit(x, y)
y_pred = gbm0.predict(x)
y_predprob = gbm0.predict_proba(x)[:, 1]
print("Accuracy:\t", metrics.accuracy_score(y, y_pred))
print("AUC Score (Train):\t", metrics.roc_auc_score(y, y_predprob))
print("Accuracy:\t", metrics.accuracy_score(ys, y_pred))
print("AUC Score (Train):\t", metrics.roc_auc_score(ys, y_predprob))

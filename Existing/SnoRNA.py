import random
import time

import cv2
import numpy
import numpy as np
import pandas as pd
from numpy import interp
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, LeaveOneOut

from Existing.ourCode import startTime

SnoRNA_similarity = pd.read_csv('newData/SnoS.csv')
disease_name = pd.read_csv('newData/disease_name.csv')
snoRNA_name = pd.read_csv('newData/snoRNA_name.csv')
known_association = pd.read_csv('newData/SnoD.csv')
disease_similarity = pd.read_csv('newData/DS.csv')




disease_semantic_similarity = numpy.zeros((60, 60))
adjacency_matrix = numpy.zeros((571, 60))
snoRNA_functional_similarity = numpy.zeros((571, 571))

#csv to array disease_semantic_similarity
for i in range(len(disease_name)):
    for j in range(len(disease_name)):
        disease_semantic_similarity[i, j] = disease_similarity.iloc[i, j]

#csv to array adjacency_matrix
for i in range(len(snoRNA_name)):
    for j in range(len(disease_name)):
        adjacency_matrix[i, j] = known_association.iloc[i, j]



#csv to array snoRNA_functional_similarity
for i in range(len(snoRNA_name)):
    for j in range(len(snoRNA_name)):
        snoRNA_functional_similarity[i, j] = SnoRNA_similarity.iloc[i, j]


# K_mean clustering
unknown = []
known = []
for x in range(571):
    for y in range(60):
        if adjacency_matrix[x, y] == 0:
            unknown.append((x, y))
        else:
            known.append((x, y))
major = []
for z in range(len(unknown)):
    q = (disease_semantic_similarity[unknown[z][1], :].tolist()
         + snoRNA_functional_similarity[unknown[z][0],:].tolist())
    major.append(q)



print(len(major))

kmeans = KMeans(n_clusters=10, random_state=0).fit( major)
center = kmeans.cluster_centers_
center_x = []
center_y = []
for j in range(len(center)):
    center_x.append(center[j][0])
    center_y.append(center[j][1])
labels = kmeans.labels_



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

for i in range(len(labels)):  # 将所有未知关联（疾病，miRNA序列号）进行分类，分成23类并记录
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


type = [[], [], [], [], [], [], [], [], [], []]  # 23簇
mtype = [[], [], [], [], [], [], [], [], [], []]
dataSet = []
mtype1 = [[], [], [], [], [], [], [], [], [], []]

for k1 in range(len(type1_x)):
    type[0].append((type1_x[k1], type1_y[k1]))
for k2 in range(len(type2_x)):
    type[1].append((type2_x[k2], type2_y[k2]))
for k3 in range(len(type3_x)):
    type[2].append((type3_x[k3], type3_y[k3]))
for k4 in range(len(type4_x)):
    type[3].append((type4_x[k4], type4_y[k4]))
for k5 in range(len(type5_x)):
    type[4].append((type5_x[k5], type5_y[k5]))
for k6 in range(len(type6_x)):
    type[5].append((type6_x[k6], type6_y[k6]))
for k7 in range(len(type7_x)):
    type[6].append((type7_x[k7], type7_y[k7]))
for k8 in range(len(type8_x)):
    type[7].append((type8_x[k8], type8_y[k8]))
for k9 in range(len(type9_x)):
    type[8].append((type9_x[k9], type9_y[k9]))
for k10 in range(len(type10_x)):
    type[9].append((type10_x[k10], type10_y[k10]))

for t in range(10):
    mtype1[t] = random.sample(type[t], int((len(type[t]) / len(labels)) * len(known)))

for m2 in range(60):
    for n2 in range(571):
        for z2 in range(10):
            if (m2, n2) in mtype1[z2]:
                dataSet.append((m2, n2))  # Store the randomly extracted 23X240 samples in the dataSet
for m3 in range(571):
    for n3 in range(60):
        if adjacency_matrix[m3, n3] == 1:  # dataset存的是（疾病号，mirna号）
            dataSet.append((m3, n3))  # Combine Major and Minor into training samples containing 10,950 samples

length = len(dataSet)
# Decision tree
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
D = numpy.ones(length) * 1.0 / length # Initialize the weight of the training sample
for xx in dataSet:  # for example in dataset:for循环，在数据dataset（可以是列表、元组、集合、字典等）中 逐个取值存入 变量 example中，然后运行循环体
    q = disease_semantic_similarity[xx[1], :].tolist() + snoRNA_functional_similarity[xx[0], :].tolist()  # dataset存的是（疾病号，mirna号）  SD疾病综合相似性矩阵   X[1,:] 取第一行的所有列数据
    x.append(q)
    if (xx[0], xx[1]) in known:
        y.append(1)  # 标签  正样本为1
    else:
        y.append(0)
ys = numpy.array(y)
xs = numpy.array(x)


GBDT=GradientBoostingClassifier(n_estimators = 12,max_depth=5,min_samples_leaf=13)
GBDT.fit(xs,ys)
OHE = OneHotEncoder()
OHE.fit(GBDT.apply(xs)[:, :, 0])#model.apply(X_train)返回训练数据X_train在训练好的模型里每棵树中所处的叶子节点的位置（索引）
LR = LogisticRegression()
tprs = []
aucs = []
mean_fpr = np.linspace(0,1,100)


# Assuming xs is your input data and ys is your target variable
X_train, X_test, y_train, y_test = train_test_split(xs, ys, test_size=0.2, random_state=42)

# Now, proceed with your model fitting and evaluation using these splits
# for example:
LR.fit(OHE.transform(GBDT.apply(X_train)[:, :, 0]), y_train)
probas_ = LR.predict_proba(OHE.transform(GBDT.apply(X_test)[:, :, 0]))
fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
tprs.append(interp(mean_fpr,fpr,tpr))
tprs[-1][0] = 0.0
roc_auc = auc(fpr, tpr)
aucs.append(roc_auc)

# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_curve, auc
# from scipy import interp
#
# # Assuming you have defined tprs, aucs, mean_fpr previously
# mean_tpr = np.mean(tprs, axis=0)
# mean_auc = auc(mean_fpr, mean_tpr)
#
# plt.figure(figsize=(8, 6))
# plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.2f})')
# plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC)')
# plt.legend()
# plt.show()
#
#
# from sklearn.metrics import precision_recall_curve
# import matplotlib.pyplot as plt
#
# # Assuming you have predictions and true labels
# # Replace predictions and true labels with your actual values
# predictions = LR.predict_proba(OHE.transform(GBDT.apply(X_test)[:, :, 0]))[:, 1]
# true_labels = y_test
#
# precision, recall, _ = precision_recall_curve(true_labels, predictions)
#
# plt.figure(figsize=(8, 6))
# plt.plot(recall, precision, color='b', label='Precision-Recall curve')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall Curve')
# plt.legend()
# plt.show()

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


gbm0 = GradientBoostingClassifier(n_estimators = 12,max_depth=5,min_samples_leaf=13)    #range(0, 30, 5)  # 从 0 开始到 30步长为 5
gbm0.fit(x,y)
y_pred = gbm0.predict(x)# 返回预测标签
y_predprob = gbm0.predict_proba(x)[:,1] #predict_proba# 返回预测属于某标签的概率并且每一行的概率和为1。
print("Accuracy : %.4g" ,metrics.accuracy_score(y.values, y_pred))
print ("AUC Score (Train): %f" ,metrics.roc_auc_score(y, y_predprob))
print('Completed.Took %f s.' % (time.time() - startTime))
print ("Accuracy :" ,metrics.accuracy_score(ys, y_pred))
print ("AUC Score (Train):", metrics.roc_auc_score(ys, y_predprob))
# cv =LeaveOneOut(len(ys))






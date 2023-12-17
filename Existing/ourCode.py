# -*- coding: utf-8 -*-
import random

import pandas as pd
import xlrd
import numpy
import time
import numpy as np
import numpy.linalg as LA
from sklearn.cluster import KMeans

startTime = time.time()

xlrd.xlsx.ensure_elementtree_imported(False, None)

xlrd.xlsx.Element_has_iter = True

a = open('data/Known_disease_miRNA_association_number.xlsx')
b = open('data/Disease_semantic_similarity_matrix_model_1.xlsx')
c = open('data/Disease_semantic_similarity_matrix_model_2.xlsx')
d = open('data/Disease_semantic_similarity_weighting_matrix.xlsx')
e = open('data/miRNA_functional_similarity_matrix.xlsx')
f = open('data/miRNA_functional_similarity_weighting_matrix.xlsx')
g = open('data/miRNA_number.xlsx')
h = open('data/disease_number.xlsx')


disease_semantic_similarity = numpy.zeros((383, 383))
adjacency_matrix = numpy.zeros((383, 495))
kernel_disease_similarity = numpy.zeros((383, 383))
disease_integrated_similarity = numpy.zeros((383, 383))
disease_weighted_matrix = numpy.zeros((383, 383))
miRNA_weighted_matrix = numpy.zeros((495, 495))
miRNA_functional_similarity = numpy.zeros((495, 495))
microRNA_integrated_similarity = numpy.zeros((495, 495))
kernel_microRNA_similarity = numpy.zeros((495, 495))

xlsx1 = xlrd.open_workbook('data/Disease_semantic_similarity_matrix_model_1.xlsx')
xlsx2 = xlrd.open_workbook('data/Disease_semantic_similarity_matrix_model_2.xlsx')  # 打开Excel文件读取数据
sheet1 = xlsx1.sheets()[0]
sheet2 = xlsx2.sheets()[0]
for i in range(383):
    for j in range(383):
        s1 = sheet1.row_values(i)
        s2 = sheet2.row_values(i)
        m = s1[j]
        n = s2[j]
        disease_semantic_similarity[i, j] = float(m + n) / 2  # Get disease semantic similarity matrix SD


xlsx3 = xlrd.open_workbook('data/Known_disease_miRNA_association_number.xlsx')
sheet3 = xlsx3.sheets()[0]
for i in range(5430):
    s3 = sheet3.row_values(i)
    m = int(s3[0])
    n = int(s3[1])
    adjacency_matrix[n - 1, m - 1] = 1  # Obtain adjacency matrix adjacency_matrix



xlsx4 = xlrd.open_workbook('data/Disease_semantic_similarity_weighting_matrix.xlsx')
sheet4 = xlsx4.sheets()[0]
for i in range(383):
    for j in range(383):
        s4 = sheet4.row_values(i)
        disease_weighted_matrix[i, j] = s4[j]  # Get disease semantic weighting matrix DJQ


xlsx5 = xlrd.open_workbook('data/miRNA_functional_similarity_weighting_matrix.xlsx')
sheet5 = xlsx5.sheets()[0]
for i in range(495):
    for j in range(495):
        s5 = sheet5.row_values(i)
        miRNA_weighted_matrix[i, j] = s5[j]  # Get miRNA functional similarity weighting matrix MJQ



xlsx6 = xlrd.open_workbook('data/miRNA_functional_similarity_matrix.xlsx')
sheet6 = xlsx6.sheets()[0]
for i in range(495):
    for j in range(495):
        s6 = sheet6.row_values(i)
        miRNA_functional_similarity[i, j] = s6[j]  # Get miRNA functional similarity matrix miRNA_functional_similarity



adjacency_numpy_matrix = np.asmatrix(adjacency_matrix) # (convert matrix to numpy)
gamd = 383 / (LA.norm(adjacency_numpy_matrix,'fro') ** 2)
kd = np.mat(np.zeros((383, 383)))  # 创建一个零矩阵
km = np.mat(np.zeros((495, 495)))
D = adjacency_numpy_matrix * adjacency_numpy_matrix.T  # adjacency_numpy_matrix*C的转置矩阵

for i in range(383):
    for j in range(i, 383):
        kd[j, i] = np.exp(-gamd * (D[i, i] + D[j, j] - 2 * D[i, j]))  # 高斯核相似计算

kd = kd + kd.T - np.diag(np.diag(kd))  # 两次使用diag() 获得某二维矩阵的对角矩阵：
kernel_disease_similarity = np.asarray(kd)  # Obtain Gaussian interaction profile kernel similarity for disease disease_integrated_similarity




disease_integrated_similarity = np.multiply(disease_semantic_similarity, disease_weighted_matrix) + np.multiply(
    kernel_disease_similarity, (1 - disease_weighted_matrix))  # np.multiply(adjacency_matrix,B)矩阵对应元素位置相乘   疾病综合相似性
disease_integrated_similarity = np.asarray(
    disease_integrated_similarity)  # disease_weighted_matrix  is disease semantic weighting matrix
gamam = 495 / (LA.norm(adjacency_numpy_matrix, 'fro') ** 2)
E = adjacency_numpy_matrix.T * adjacency_numpy_matrix;
for i in range(495):
    for j in range(i, 495):
        km[i, j] = np.exp(-gamam * (E[i, i] + E[j, j] - 2 * E[i, j]))
km = km + km.T - np.diag(np.diag(km))
kernel_microRNA_similarity = np.asarray(km) # Obtain Gaussian interaction profile kernel similarity for miRNA



#
microRNA_integrated_similarity = np.multiply(miRNA_functional_similarity, miRNA_weighted_matrix) + np.multiply(
    kernel_microRNA_similarity, (1 - miRNA_weighted_matrix))
microRNA_integrated_similarity = np.asarray(microRNA_integrated_similarity)  # Obtain Gaussian interaction profile kernel similarity for miRNA microRNA_integrated_similarity
data = pd.DataFrame(microRNA_integrated_similarity)



# K_mean clustering
unknown = []
known = []
for x in range(383):
    for y in range(495):
        if adjacency_matrix[x, y] == 0:
            unknown.append((x, y))  # append() 方法向列表的尾部添加一个新的元素
        else:
            known.append((x, y))  # Divide the samples into Major and Minor
major = []
for z in range(len(unknown)):
    q = (disease_integrated_similarity[unknown[z][0], :].tolist()
         + microRNA_integrated_similarity[unknown[z][1],:].tolist())  # tolist()将数组或者矩阵转换成列表    X[1,:] 取第一行的所有列数据
    major.append(q)

kmeans = KMeans(n_clusters=23, random_state=0).fit( major)  # n_clusters 和random_state。其中，前者表示你打算聚类的数目，默认情况下是8。后者表示产生随机数的方法

center = kmeans.cluster_centers_  # 获取簇心 #获得训练后模型23类中心点位置坐标 为23x2的矩阵 fit()方法是对Kmeans确定类别以后的数据集进行聚类
center_x = []
center_y = []
for j in range(len(center)):
    center_x.append(center[j][0])
    center_y.append(center[j][1])
labels = kmeans.labels_  # 标注每个点的聚类结果  得到聚类后每一组数据对应的类型标签，数据为一个列表labels[1,2,3,1......]对应每一组数据的类别
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
type21_x = []
type21_y = []
type22_x = []
type22_y = []
type23_x = []
type23_y = []
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
    if labels[i] == 20:
        type21_x.append(unknown[i][0])
        type21_y.append(unknown[i][1])
    if labels[i] == 21:
        type22_x.append(unknown[i][0])
        type22_y.append(unknown[i][1])
    if labels[i] == 22:
        type23_x.append(unknown[i][0])
        type23_y.append(unknown[i][1])
type = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]  # 23簇
mtype = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
dataSet = []
mtype1 = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
# print('Completed.Took %f s.' % (time.time() - startTime))
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
for k11 in range(len(type11_x)):
    type[10].append((type11_x[k11], type11_y[k11]))
for k12 in range(len(type12_x)):
    type[11].append((type12_x[k12], type12_y[k12]))
for k13 in range(len(type13_x)):
    type[12].append((type13_x[k13], type13_y[k13]))
for k14 in range(len(type14_x)):
    type[13].append((type14_x[k14], type14_y[k14]))
for k15 in range(len(type15_x)):
    type[14].append((type15_x[k15], type15_y[k15]))
for k16 in range(len(type16_x)):
    type[15].append((type16_x[k16], type16_y[k16]))
for k17 in range(len(type17_x)):
    type[16].append((type17_x[k17], type17_y[k17]))
for k18 in range(len(type18_x)):
    type[17].append((type18_x[k18], type18_y[k18]))
for k19 in range(len(type19_x)):
    type[18].append((type19_x[k19], type19_y[k19]))
for k20 in range(len(type20_x)):
    type[19].append((type20_x[k20], type20_y[k20]))
for k21 in range(len(type21_x)):
    type[20].append((type21_x[k21], type21_y[k21]))
for k22 in range(len(type22_x)):
    type[21].append((type22_x[k22], type22_y[k22]))
for k23 in range(len(type23_x)):
    type[22].append((type23_x[k23], type23_y[k23]))  # Divide Major into 23 clusters by K-means clustering

for t in range(23):
    mtype1[t] = random.sample(type[t], int((len(type[t]) / len(labels)) * 5430))
for m2 in range(383):
    for n2 in range(495):
        for z2 in range(23):
            if (m2, n2) in mtype1[z2]:
                dataSet.append((m2, n2))  # Store the randomly extracted 23X240 samples in the dataSet
for m3 in range(383):
    for n3 in range(495):
        if adjacency_matrix[m3, n3] == 1:  # dataset存的是（疾病号，mirna号）
            dataSet.append((m3, n3))  # Combine Major and Minor into training samples containing 10,950 samples

print(len(dataSet))
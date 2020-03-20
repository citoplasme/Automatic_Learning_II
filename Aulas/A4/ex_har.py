#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd

x_train = pd.read_csv("UCI HAR Dataset/train/X_train.txt", sep = "\s+", header = None)
x_test = pd.read_csv("UCI HAR Dataset/test/X_test.txt", sep = "\s+", header = None)
print(x_train.shape, x_test.shape)

y_train = pd.read_csv("UCI HAR Dataset/train/y_train.txt", header=None, names=['ActivityID'])
y_test = pd.read_csv("UCI HAR Dataset/test/y_test.txt", header=None, names=['ActivityID'])
print(y_train.shape, y_test.shape)

subjects_tr = pd.read_csv("UCI HAR Dataset/train/subject_train.txt", header = None, names=['SubjectID'])
subjects_tr.head()
subjects_tst = pd.read_csv("UCI HAR Dataset/test/subject_test.txt", header = None, names=['SubjectID'])
print(subjects_tr.shape, subjects_tst.shape)

features = pd.read_csv("UCI HAR Dataset/features.txt", sep = " ", header = None, names=('ID','Sensor'))
print(features.shape)
features.head()

activities = pd.read_csv('UCI HAR Dataset/activity_labels.txt', sep=' ', header=None, names=('ID','Activity'))
print(activities)

## Merge datasets

subjects_all = pd.concat([subjects_tr, subjects_tst], ignore_index=True)
print(subjects_all.shape)

x_all = pd.concat([x_train, x_test], ignore_index = True)
print(x_all.shape)

y_all = y_train.append(y_test, ignore_index=True)
print(y_all.shape)

sensorNames = features['Sensor']
x_all.columns = sensorNames
x_all.head()

### HA NOMES REPETIDOS - VERIFICAR
for i in range(561):
    for j in range(i+1, 561):
        if features["Sensor"][i] == features["Sensor"][j]:
            print(i, j, features["Sensor"][i])


# Merge Subjects and sensors data frames by columns
x_all = pd.concat([x_all, subjects_all], axis=1)
print(x_all.shape)

for i in activities['ID']:
  activity = activities[activities['ID'] == i]['Activity'] 
  y_all = y_all.replace({i: activity.iloc[0]})
 
y_all.columns = ['Activity']
y_all.head()
y_all.tail()

allXy = pd.concat([x_all, y_all], axis=1)
print(allXy.shape)
allXy.to_csv("HAR_clean.csv")

## aggregate

import numpy as np

grouped = allXy.groupby (['SubjectID', 'Activity']).aggregate(np.mean)
print(grouped.shape)
grouped.head()
grouped.to_csv("HAR_grouped.csv")

## exploring

grouped.describe()

allXy.groupby("Activity").size()
grouped.groupby("Activity").size()


allXy.isnull().sum().sum()

allXy.iloc[:,:-2].mean()
allXy.iloc[:,:-2].max()
allXy.iloc[:,:-2].min()
allXy.iloc[:,:-2].std()

# PCA
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

scaled = preprocessing.scale(allXy.iloc[:,:-2])

n = 10
pca = PCA(n_components=n)
pca.fit(scaled)
X_reduced = pca.transform(scaled)
#X_r = pca.fit(allXy.iloc[:,:-2]).transform(allXy.iloc[:,:-2])

print('Var. explained: %s'% str(pca.explained_variance_ratio_))

plt.bar(range(n), pca.explained_variance_ratio_*100)
plt.xticks(range(n), ['PC'+str(i) for i in range(1,n+1)])
plt.title("Explained variance")
plt.ylabel("Percentage")
plt.show()

 # score plot
for act in allXy['Activity'].unique():
    sp = allXy.index[allXy['Activity']==act]-1
    plt.plot(X_reduced[sp,0],X_reduced[sp,1], 'o' , label=act)
plt.title("PCA")
plt.legend(loc='best', shadow=False)
plt.show()

## K-means

from sklearn.cluster import KMeans

k=6
kmeans_har = KMeans(n_clusters=k, max_iter=1000)
kmeans_har.fit(scaled)
labels = kmeans_har.labels_
centroids = kmeans_har.cluster_centers_

pd.crosstab(labels,allXy["Activity"], rownames=['clusters'] )

# HC

from scipy.cluster.hierarchy import dendrogram, linkage

grouped_sc = preprocessing.scale(grouped.iloc[:,2:])

Z = linkage(grouped_sc, method='single', metric='euclidean')

plt.figure(figsize=(25, 10))
dendrogram(
    Z,
    labels=list(grouped.index.get_level_values(1)),
    leaf_rotation=90., 
    leaf_font_size=8.
)
plt.title('Hierarchical Clustering Dendrogram')
plt.ylabel('distance')

label_colors = {'STANDING': 'b', "WALKING_UPSTAIRS": "m", "LAYING": 'g', 'SITTING': 'c', "WALKING" : "y", "WALKING_DOWNSTAIRS": "r"}
ax = plt.gca()
xlbls = ax.get_xmajorticklabels()
for lbl in xlbls:
    lbl.set_color(label_colors[lbl.get_text()])
plt.show()







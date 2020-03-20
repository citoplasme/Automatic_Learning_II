#!/usr/bin/python3

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


# -------------------------- Attribute Selection ------------------------

# por variabilidade
from sklearn.feature_selection import VarianceThreshold 
sel = VarianceThreshold(threshold=0.05)
filt_var = sel.fit_transform(allXy.iloc[:,:-2]) 
print(filt_var .shape)

# Testes Estatisticos Univariados 
from sklearn.feature_selection import SelectPercentile, f_classif 
selector = SelectPercentile(f_classif, percentile=30)
filt_univ = selector.fit_transform(allXy.iloc[:,:-2], allXy.iloc[:,-1]) 
print(filt_univ .shape)

# Remover atributos altamente correlacionados
corr_matrix = allXy.iloc[:,:-2].corr() 
drop_cols = []
threshold_cor = 0.9
for i in range(len(corr_matrix.columns) - 1):
	for j in range(i+1,len(corr_matrix.columns)): 
		if j not in drop_cols:
			item = corr_matrix.iloc[i:(i+1), j:(j+1)] 
			val = item.values
			if abs(val) >= threshold_cor:
				drop_cols.append(j) 
print(drop_cols)

keep = []
for i in range(len(corr_matrix.columns)):
	if i not in drop_cols: 
		keep.append(i)
orig = allXy.iloc[:,:-2] 
filt_cor = orig.iloc[:,keep] 
print(filt_cor.shape)

# -------------------------- Divisão dos Dados ------------------------
from sklearn.model_selection import train_test_split

X_tr, X_ts, y_tr, y_ts = train_test_split(allXy.iloc[:,:-2], allXy.iloc[:,-1], test_size=0.3)
print(X_tr.shape, y_tr.shape) 
print(X_ts.shape, y_ts.shape)
print(y_tr.value_counts()/len(y_tr)) 
print(y_ts.value_counts()/len(y_ts))

# -------------------------- Modelos Baseline ------------------------
# KNN

from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier() 
knn_model.fit(X_tr, y_tr) 
knn_model.score(X_ts, y_ts)

# SVM, 

from sklearn import svm
svm_model = svm.SVC(gamma=0.001, C=10.) 
svm_model.fit(X_tr, y_tr) 
svm_model.score(X_ts, y_ts)


# Ensembles
	# Random Forest
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100) 
rf_model.fit(X_tr, y_tr)
rf_model.score(X_ts, y_ts)

	# bagging
from sklearn.ensemble import BaggingClassifier 
from sklearn import tree
bag_model = BaggingClassifier(tree.DecisionTreeClassifier()) 
bag_model.fit(X_tr, y_tr)
bag_model.score(X_ts, y_ts)

# -------------------- Testar dados com feature selection ----------------

# Variabilidade
X_tr_f, X_ts_f, y_tr_f, y_ts_f = train_test_split(filt_var, allXy.iloc[:,-1], test_size= 0.3)
svm_model.fit(X_tr_f, y_tr_f) 
svm_model.score(X_ts_f, y_ts_f)

# Correlações
X_tr_f2, X_ts_f2, y_tr_f2, y_ts_f2 = train_test_split(filt_cor, allXy.iloc[:,-1], test_size= 0.3)
svm_model.fit(X_tr_f2, y_tr_f2) 
svm_model.score(X_ts_f2, y_ts_f2)

# ------------------------ Otimização de Hiperparâmetros ------------------

# Grid search; SVM com kernel Gaussiano

from sklearn.model_selection import GridSearchCV 
parameters = {'C':[1, 10, 100], 'gamma':[0.01, 0.001]}
svm_model_d = svm.SVC()
opt_model_d = GridSearchCV(svm_model_d, parameters)
opt_model_d.fit(X_tr, y_tr)
print (opt_model_d.best_estimator_)
opt_model_d.score(X_ts, y_ts)
 

# Modelo Final 
opt_model_d.fit(allXy.iloc[:,:-2], allXy.iloc[:,-1])




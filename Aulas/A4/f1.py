from sklearn import datasets 
from sklearn import linear_model

diabetes = datasets.load_diabetes() 
Xd_train = diabetes.data[:-20] 
Xd_test = diabetes.data[-20:] 
yd_train = diabetes.target[:-20] 
yd_test = diabetes.target[-20:]

linmodel = linear_model.LinearRegression()
linmodel = linmodel.fit(Xd_train, yd_train) 
print("Valores previstos: " , linmodel.predict(Xd_test))

ridge = linear_model.Ridge(alpha=.1)
ridge = ridge.fit(Xd_train, yd_train)
print("Valores previstos: " , ridge.predict(Xd_test))

# Lasso + Rifge
lasso = linear_model.Lasso()
lasso = lasso.fit(Xd_train, yd_train)
print("Valores previstos: " , lasso.predict(Xd_test))


# SVMs
from sklearn import svm
from sklearn.model_selection import cross_val_score, train_test_split

# Classification
iris = datasets.load_iris() 
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size= 0.3)
print(X_train.shape, y_train.shape) 
print(X_test.shape, y_test.shape)
svm_model = svm.SVC(kernel='linear', C=1) 
svm_model.fit(X_train, y_train) 
print(svm_model.score(X_test, y_test))

# Regression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
svm_reg = SVR(gamma = "auto")
svm_reg = svm_reg.fit(Xd_train, yd_train) 
pred_svm = lasso.predict(Xd_test) 
print("Valores previstos: " , pred_svm)
mse = mean_squared_error(yd_test, pred_svm) 
print("MSE: %.1f" % mse)


# Cross Validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(svm_model, iris.data, iris.target, cv = 5) 
print(scores)
print(scores.mean())

scores_f1 = cross_val_score(svm_model,
iris.data, y = iris.target, scoring = "f1_weighted", cv = 5)
print(scores_f1) 
print(scores_f1.mean())

# Leave One Out
from sklearn.model_selection import LeaveOneOut
loo_cv = LeaveOneOut()
scores_loo = cross_val_score(svm_model, iris.data, iris.target, cv=loo_cv)
print(scores_loo.mean())


# Attribute Selection

# Remove attributes that dont change a lot (below threshold)
from sklearn import datasets, svm
from sklearn.feature_selection import VarianceThreshold 
from sklearn.model_selection import cross_val_score
digits = datasets.load_digits()
print (digits.data.shape)
sel = VarianceThreshold(threshold=20)
filt = sel.fit_transform(digits.data)
print (filt.shape)
svm_model = svm.SVC(gamma=0.001, C=100.)
scores= cross_val_score(svm_model, digits.data, digits.target, cv= 10) 
print (scores.mean())
scores_vt= cross_val_score(svm_model, filt, digits.target, cv= 10) 
print (scores_vt.mean())


# Keep attributes better than p-value
from sklearn.feature_selection import SelectKBest, chi2, f_classif
filt_kb = SelectKBest(chi2, k=32).fit_transform(digits.data, digits.target) 
print (filt_kb.shape)
scores_kb = cross_val_score(svm_model, filt_kb, digits.target, cv = 10) 
print (scores_kb.mean())
filt_kb2 = SelectKBest(f_classif, k=32).fit_transform(digits.data, digits.target) 
scores_kb2 = cross_val_score(svm_model, filt_kb2, digits.target, cv = 10) 
print (scores_kb2.mean())


# Recursive feature elimination (Wrapper)
from sklearn.feature_selection import RFE
svm_model = svm.SVC(kernel = "linear", C=100.)
rfe = RFE(estimator=svm_model, n_features_to_select=32, step=1) 
scores_rfe = cross_val_score(rfe, digits.data, digits.target, cv = 10) 
print(scores_rfe.mean())


# Procura em grelha de parâmetros de SVMs (com validação cruzada na estimação do erro)
from sklearn import svm, datasets
from sklearn.model_selection import cross_val_score, GridSearchCV
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 3, 10, 100], 'gamma':[0.01, 0.001]}
svm_model_d = svm.SVC( )
opt_model_d = GridSearchCV(svm_model_d, parameters) 
opt_model_d.fit(digits.data, digits.target)
print (opt_model_d.best_estimator_)
scores_gs = cross_val_score(opt_model_d, digits.data, digits.target, cv = 5) 
print(scores_gs.mean())


# Procura aleatória (com validação cruzada na estimação do erro)
from sklearn.model_selection import RandomizedSearchCV 
from sklearn.ensemble import RandomForestClassifier

# Procura aleatória (com validação cruzada na estimação do erro)
import numpy as np
def report(results, n_top=3): 
	for i in range(1, n_top + 1):
		candidates = np.flatnonzero(results['rank_test_score'] == i) 
		for candidate in candidates:
			print("Model with rank: {0}".format(i))
			print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
				results['mean_test_score'][candidate],
				results['std_test_score'][candidate]))
			print("Parameters: {0}".format(results['params'][candidate])) 
			print("")


rf_model = RandomForestClassifier(n_estimators=100) 
param_dist = {"max_depth": [2, 3, None], "max_features": [2,4,6], "min_samples_split": [2,4,6], "min_samples_leaf": [2,4,6], "bootstrap": [True, False], "criterion": ["gini", "entropy"]} 
rand_search = RandomizedSearchCV(rf_model, param_distributions=param_dist, n_iter=20, cv = 5) 
rand_search.fit(digits.data, digits.target)
print (rand_search.best_estimator_)
report(rand_search.cv_results_)
scores_rs = cross_val_score(rand_search, digits.data, digits.target, cv = 5) 
print (scores_rs.mean())



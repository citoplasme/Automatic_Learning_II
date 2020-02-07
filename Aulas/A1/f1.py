from sklearn import datasets 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import cross_val_score 
from sklearn import linear_model

# -----------------------------------------------------------------
# ------------------------- Datasets ------------------------------
# -----------------------------------------------------------------

iris = datasets.load_iris() 
print(iris.data) 
print(iris.target) 
print(iris.data.shape) 
print(np.unique(iris.target))

digits = datasets.load_digits() 
print(digits.data) 
print(digits.target) 
print(digits.data.shape)


plt.figure(1, figsize=(3, 3)) 
plt.imshow(digits.images[0], cmap=plt.cm.gray_r, interpolation='nearest') 
plt.show()
print(digits.target[0])

# -----------------------------------------------------------------
# ------------------------- KNN -----------------------------------
# -----------------------------------------------------------------

indices = np.random.permutation(len(iris.data)) 
train_in = iris.data[indices[:-10]]
train_out = iris.target[indices[:-10]]
test_in = iris.data[indices[-10:]]
test_out = iris.target[indices[-10:]]

knn = KNeighborsClassifier() # K = 1 default
print(knn.fit(train_in, train_out))
print("Valores previstos: ", knn.predict(test_in)) 
print("Valores reais: ", test_out)

# -----------------------------------------------------------------
# ------------------------- Log -----------------------------------
# -----------------------------------------------------------------

logistic = linear_model.LogisticRegression(solver = "lbfgs", multi_class = "auto")
logistic = logistic.fit(train_in, train_out) 
print(logistic)
print("Valores previstos: ", logistic.predict(test_in)) 
print("Valores reais: ", test_out)

# -----------------------------------------------------------------
# ------------------------- Linear --------------------------------
# -----------------------------------------------------------------

diabetes = datasets.load_diabetes() 
X_train = diabetes.data[:-20]
X_test = diabetes.data[-20:]
y_train = diabetes.target[:-20] 
y_test = diabetes.target[-20:]


regr_model = linear_model.LinearRegression() 
regr_model = regr_model.fit(X_train, y_train) 
print(regr_model)
print("Valores previstos: ", regr_model.predict(X_test)) 
print("Valores reais: ", y_test)


# -----------------------------------------------------------------
# ------------------------- Error ---------------------------------
# -----------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size= 0.3) 
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

log_model = linear_model.LogisticRegression(solver = "lbfgs", multi_class = "auto", max_iter = 1000)
log_model = log_model.fit(X_train, y_train) 
print(log_model.score(X_test, y_test))


# -----------------------------------------------------------------
# --------------------- Cross Validation --------------------------
# -----------------------------------------------------------------

scores = cross_val_score(log_model, iris.data, iris.target, cv = 5) 
print(scores)
print(scores.mean())

print("Funcao scoring: F1")
scores_f1 = cross_val_score(log_model,
iris.data, y = iris.target, scoring = "f1_weighted", cv = 5) 
print(scores_f1)
print(scores_f1.mean())


# -----------------------------------------------------------------
# ---------------------------- R^2 --------------------------------
# -----------------------------------------------------------------

regr_model = linear_model.LinearRegression() 
scores_r2 = cross_val_score(regr_model, diabetes.data, y = diabetes.target, scoring = "r2", cv = 5) 
print(scores_r2)
print(scores_r2.mean())


# -----------------------------------------------------------------
# ------------------------ MAD & MSE ------------------------------
# -----------------------------------------------------------------

scores_mse = cross_val_score(regr_model, diabetes.data, diabetes.target, scoring= "neg_mean_squared_error", cv= 5)
print (scores_mse)
scores_mad = cross_val_score(regr_model, diabetes.data, diabetes.target, scoring = "neg_mean_absolute_error", cv= 5)
print (scores_mad)




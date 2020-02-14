from sklearn import datasets 
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import cross_val_predict 
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.model_selection import cross_val_score 
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
# import seaborn as sns
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC

# --------------------------------------------------------------
# ------------------------- Test Set ---------------------------
# --------------------------------------------------------------

iris = datasets.load_iris()
indices = np.random.permutation(len(iris.data)) 
train_in = iris.data[indices[:-10]]
train_out = iris.target[indices[:-10]]
test_in = iris.data[indices[-10:]]
test_out = iris.target[indices[-10:]]

# --------------------------------------------------------------
# ---------------------- Decision Trees ------------------------
# --------------------------------------------------------------

tree_model = tree.DecisionTreeClassifier() 
tree_model = tree_model.fit(train_in, train_out) 
print(tree_model)
print("Valores previstos: ", tree_model.predict(test_in)) 
print("Valores reais: ", test_out)

# --------------------------------------------------------------
# ------------------------- Tree Plot --------------------------
# --------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state = 123)
tree_clf = DecisionTreeClassifier(max_depth = 2) 
tree_clf.fit(X_train, y_train)
graph = export_graphviz(tree_clf, out_file="iris_tree.dot", feature_names=iris.feature_names, class_names = iris.target_names, rounded=True, filled = True)


# --------------------------------------------------------------
# ---------------------- Tree Prediction -----------------------
# --------------------------------------------------------------

dt_model = DecisionTreeClassifier()
scores = cross_val_score(dt_model, iris.data, iris.target, cv = 5) 
print(scores.mean())
predictions = cross_val_predict(dt_model, iris.data, iris.target, cv = 5) 
conf_mat = confusion_matrix(iris.target, predictions)
conf_mat = pd.DataFrame(conf_mat)
conf_mat.index.name = 'Actual'
conf_mat.columns.name = 'Predicted' 
print(conf_mat)


# --------------------------------------------------------------
# ---------------------- Cross Validation ----------------------
# --------------------------------------------------------------

dt_model1 = DecisionTreeClassifier(criterion = "gini") 
dt_model2 = DecisionTreeClassifier(criterion = "entropy")
scores_dt1 = cross_val_score(dt_model1, iris.data, iris.target, cv = 5) 
scores_dt2 = cross_val_score(dt_model1, iris.data, iris.target, cv = 5) 
print(scores_dt1.mean(), scores_dt2.mean())


# --------------------------------------------------------------
# ----------------------- Tree Regressor -----------------------
# --------------------------------------------------------------

diabetes = datasets.load_diabetes()

regressor = DecisionTreeRegressor()
scores_r2 = cross_val_score(regressor, diabetes.data, diabetes.target, cv=10) 
print(scores_r2.mean())
scores_mse = cross_val_score(regressor, diabetes.data, diabetes.target, cv=10, scoring = "neg_mean_absolute_error")
print(np.sqrt(-1*scores_mse.mean()))




# --------------------------------------------------------------
# ------------------------- Ensembles --------------------------
# --------------------------------------------------------------


# --------------------------------------------------------------
# -------------------------- Bagging ---------------------------
# --------------------------------------------------------------
bagged_model = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
scores_bag = cross_val_score(bagged_model, iris.data, iris.target, cv = 5) 
print (scores_bag)

bagged_model2 = BaggingClassifier(tree.DecisionTreeClassifier(), max_samples=0.5, max_features=0.5) 
scores_bag2 = cross_val_score(bagged_model2, iris.data, iris.target, cv = 5)
print(scores_bag2)

# --------------------------------------------------------------
# ------------------------- Boosting ---------------------------
# --------------------------------------------------------------

ada_tree = AdaBoostClassifier(n_estimators=100)
scores_ada = cross_val_score(ada_tree, iris.data, iris.target, cv = 5)
print (scores_ada)
print (scores_ada.mean())


# --------------------------------------------------------------
# -------------------- Gradient Boosting -----------------------
# --------------------------------------------------------------

X_train = diabetes.data[:-20] 
X_test = diabetes.data[-20:] 
y_train = diabetes.target[:-20] 
y_test = diabetes.target[-20:]


params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2, 'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params) 
clf.fit(X_train, y_train)
mse = mean_squared_error(y_test, clf.predict(X_test)) 
print("MSE: %.1f" % mse)
scores_mse = cross_val_score(estimator = clf, X= diabetes.data, scoring = "r2", y = diabetes.target, cv = 5) # score_func = mean_absolute_error,
print(scores_mse) 
print(scores_mse.mean())


# --------------------------------------------------------------
# ---------------------- Random Forest -------------------------
# --------------------------------------------------------------

rf_model = RandomForestClassifier(n_estimators=100)
scores_rf = cross_val_score(rf_model, iris.data, iris.target, cv = 5) 
print(scores_rf.mean())

for n_estimators in [50, 100, 200, 500]:
	rf_clf = RandomForestClassifier(n_estimators = n_estimators)
	rf_scores = cross_val_score(rf_clf, iris.data, iris.target, scoring="accuracy", cv = 10) 
	print("Num estimators: %.4f Accuracy: %.4f" %(n_estimators, rf_scores.mean()))


# --------------------------------------------------------------
# ------------------- Variable Importance ----------------------
# --------------------------------------------------------------

rf_model = rf_model.fit(iris.data, iris.target) 
importances = list(rf_model.feature_importances_)

# List of tuples with variable and importance 
feature_importances = [(feature, round(importance, 2))
	for feature, importance in zip(iris.feature_names, importances)]

# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# Print out the feature and importances
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances] 

# plot result
# sns.barplot(x=importances, y = iris.feature_names,label="Feature Importance")



# --------------------------------------------------------------
# -------------------- Voting Classifier -----------------------
# --------------------------------------------------------------

clf1 = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=4000) 
clf2 = RandomForestClassifier(n_estimators=100)
clf3 = GaussianNB()
eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
for clf, label in zip([clf1, clf2, clf3, eclf], ['Logistic', 'RF’, NB', 'Ensemble']): 
	scores = cross_val_score(clf, iris.data, iris.target, cv=5, scoring='accuracy') 
	print("Accur.: %0.2f (std %0.2f) [%s]" % (scores.mean(), scores.std(), label))

# --------------------------------------------------------------
# ---------------- Voting Classifier Weighted ------------------
# --------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size= 0.3)
clf4 = DecisionTreeClassifier(max_depth=4)
clf5 = KNeighborsClassifier(n_neighbors=5)
clf6 = SVC(gamma='scale', kernel='rbf', probability=True)
eclf = VotingClassifier(estimators=[('dt', clf4), ('knn', clf5), ('svc', clf6)], voting='soft', weights=[1, 2, 3])
eclf = eclf.fit(X_train, y_train) 
print(eclf.score(X_test, y_test))



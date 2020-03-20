# Classificação - XNOR
from sklearn.neural_network import MLPClassifier

X = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]] 
y = [1, 0, 0, 1]
mlp = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(6,) ) 
mlp.fit(X, y)
preds = mlp.predict([[0., 0.], [0., 1.], [1., 0.], [1., 1.]]) 
print(preds)

# Classificação - digits
from sklearn.model_selection import cross_val_score 
from sklearn import datasets

digits = datasets.load_digits()
mlp_dig = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(50,))
scores = cross_val_score(mlp_dig, digits.data, digits.target, cv = 5) 
print(scores.mean())

# Classificação – digits com standardização dos dados
from sklearn.preprocessing import StandardScaler

mlp_dig = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(50,))
scaler = StandardScaler() 
scaler.fit(digits.data)
scaled_digits = scaler.transform(digits.data)
scores = cross_val_score(mlp_dig, scaled_digits, digits.target, cv = 5) 
print(scores.mean())

# Regressão
from sklearn.neural_network import MLPRegressor

diabetes = datasets.load_diabetes()
mlp_diab = MLPRegressor(solver = "lbfgs", hidden_layer_sizes=(20,) )
scores = cross_val_score(mlp_diab, diabetes.data, diabetes.target, cv = 5) 
print(scores.mean())
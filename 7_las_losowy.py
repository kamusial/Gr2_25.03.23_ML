import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_decision_regions

df = pd.read_csv("iris.csv")
print(df["class"].value_counts())

species = {
    "Iris-setosa":0, "Iris-versicolor":1, "Iris-virginica":2
}
df["class_value"] = df["class"].map(species)
print(df)

# X = df[ ["sepallength", "sepalwidth"] ]   #najpierw te kontrowersyjne
# X = df[ ["petallength", "petalwidth"] ]
X = df.iloc[: , :4] # wszystkie 4 cechy
y = df.class_value

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
# model = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=0)
# model.fit(X_train, y_train)
# print(model.score(X_test, y_test))
# print(pd.DataFrame( confusion_matrix(y_test, model.predict(X_test)) ))

#dalej - wyświetlić i zbadać krawędzie decyzyjne (2 cechy)
# plot_decision_regions(X_train.values, y_train.values, model)
# plt.show()

from sklearn.model_selection import GridSearchCV
model = RandomForestClassifier()
params = {
    'n_estimators' : range(20, 101, 20),
    "max_depth" : range(2,10),
    "max_features" : range(1, X_train.shape[1]+1,1),
    "min_samples_split" : range(1, 6),
    "random_state" : [0]
}
grid = GridSearchCV(model, params, scoring="accuracy", cv=10, verbose=1)
grid.fit(X_train, y_train)
print(grid.best_params_)
print(grid.best_score_)
print(grid.best_estimator_)
print(pd.DataFrame(grid.best_estimator_.feature_importances_ , X.columns).sort_values(0, ascending=False))
y_pred = grid.best_estimator_.predict(X_test)
print(pd.DataFrame( confusion_matrix(y_test,y_pred) ))
# plot_decision_regions(X_train.values, y_train.values, grid.best_estimator_)
# plt.show()

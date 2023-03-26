import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

df = pd.read_csv('heart.csv', comment='#')
print(df.head(10).to_string())
print(df.target.value_counts())   #rozkład wartości kolumny "target"

X = df.iloc[ : , : -1]    #wszystkie wiersze, wszystkie kolumny bez ostatniej
y = df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

def drzewko(max_depth):
    print('\nMax depth wynosi',max_depth)
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))  #sprawdź model
    y_policzony_modelem = model.predict(X_test)
    print(pd.DataFrame(confusion_matrix(y_test, y_policzony_modelem)))


drzewko(2)
drzewko(3)
drzewko(5)
drzewko(7)
drzewko(9)
drzewko(11)

from sklearn.model_selection import GridSearchCV

model = DecisionTreeClassifier()
params = {
    'max_depth': range(2, 14),    #liczby całkowite od 3 do 13
    'max_features': range(3, X_train.shape[1]+1, 2),
    'min_samples_split': [2, 4, 6],
    'random_state': [0]
}
grid = GridSearchCV(model, params, scoring='accuracy', cv=10, verbose=1)
grid.fit(X_train, y_train)
print(grid.best_params_)
print(grid.best_score_)
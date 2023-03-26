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


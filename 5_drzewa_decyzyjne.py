import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("iris.csv")
print(df["class"].value_counts())
species = {
    "Iris-setosa":0, "Iris-versicolor":1, "Iris-virginica":2
}
df["class_value"] = df["class"].map(species)

# sns.scatterplot(data=df, x='sepallength', y='sepalwidth', hue='class' )
# plt.title('Sepal')
# plt.show()
# sns.scatterplot(data=df, x='petallength', y='petalwidth', hue='class' )
# plt.title('Petal')
# plt.show()
print(df.columns)
# sns.heatmap(    df.iloc[ : , : 4 ].corr(), annot=True          )
# plt.show()

#X = df[ ['sepallength', 'sepalwidth' ]    ]   #pierwszy liść - sepal
X = df[ ['petallength', 'petalwidth' ]    ]    #drugi liść - petal
y = df.class_value

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=20, random_state=0)
model.fit(X, y)

#granice decyzyjne
from mlxtend.plotting import plot_decision_regions
plot_decision_regions(X.values, y.values, model)
plt.show()

#wyrysowane drzewo decyzyjne
# from dtreeplt import dtreeplt
# dtree = dtreeplt(model=model, feature_names=X.columns, target_names=["setosa","versicolor","virginica"])
# dtree.view()
# plt.show()
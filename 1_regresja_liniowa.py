import pandas as pd    #'as'  alias
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('weight-height.csv')
#print(type(df))
#print(df)
print(df.head(5))  #5 wierszy
#print(df.Height) #dana kolumna
print(df.Gender.value_counts())
df.Height *= 2.54
df.Weight /= 2.2
print(df.head(5))
# gender, height - zmiennie niezależne -> weight - zmienna zależna
# sns.displot(df.Weight)  # kobiety i mężczyźni razem
# sns.displot(df.query("Gender=='Male'").Weight)
# sns.displot(df.query("Gender=='Female'").Weight)
# plt.show()

#gender, dana nienumeryczna
df = pd.get_dummies(df)  #zamienia dane niemeryczne, na numeryczne
del(df["Gender_Male"])   #usuń kolumnę
df.rename(columns={'Gender_Female': 'Gender'}, inplace=True)
print(df.head(5))
#dane na stole

model = LinearRegression()   #wybieram algorytm
model.fit(df[ ['Height', 'Gender'] ], df['Weight'] )   #policz
print(model.coef_, model.intercept_)
print('wzór: Height * ',model.coef_[0], '+ Gender * ',model.coef_[1],' = Weight')

#Własna formuła
gender = 0  #male
height = 192
weight = model.intercept_ + model.coef_[0] * height + model.coef_[1] * gender
print(weight)
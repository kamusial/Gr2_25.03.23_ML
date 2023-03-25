import pandas as pd    #'as'  alias
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
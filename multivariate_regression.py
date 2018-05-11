# Multivariate Regression - Predicting Car Prices
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler


df = pd.read_excel('C:/Users/Dilpreet Singh/Desktop/python-machine-learning/cars.xls')
scale = StandardScaler()
X = df[['Mileage','Cylinder','Doors']]
y = df['Price']
X[['Mileage','Cylinder','Doors']] = scale.fit_transform(X[['Mileage','Cylinder','Doors']].as_matrix())
est = sm.OLS(y, X).fit()
print(est.summary())

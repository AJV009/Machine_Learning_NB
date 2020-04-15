
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#importing the dataset
dataset=pd.read_csv('position_salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#Fitting multiple regression to to the training set
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#fitting polynomial regression to datasheet
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)


#visualising the training set results
plt.scatter(X ,y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('sal vs exp')
plt.show()

#visualising the training set results
plt.scatter(X ,y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('sal vs exp')
plt.show()
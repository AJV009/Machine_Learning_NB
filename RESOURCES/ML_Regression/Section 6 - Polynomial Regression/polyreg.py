# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#fitting linear regression model 
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)


#fitting polynomial regression model
from sklearn.preprocessing import PolynomialFeatures
poly_reg =PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly , y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

#visualising the linear Regression results
plt.scatter(X, y, color ='red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('tuth or bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('salary')
plt.show()
#visualising the polynomial Regression results
X_grid = np.arrange(min(X), max(X), 0.1)
X_grid =  X_grid.reshape((len(X_grid),1))
plt.scatter(X, y, color ='red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('tuth or bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('salary')
plt.show()

#predicting results with linear reg
lin_reg.predict(6.5)

#predicting with polynomial regression (bestone)
lin_reg_2.predict(poly_reg.fit_transform(6.5))
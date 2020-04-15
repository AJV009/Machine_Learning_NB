import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#importing the dataset
dataset=pd.read_csv('test1.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values
#splitting the dataset in to training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
#fitting simple regression to training section
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
#Predicting the test results
y_pred = regressor.predict(X_test)
#visualising the training set results
plt.scatter(X_train ,y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('sal vs exp')
plt.show()
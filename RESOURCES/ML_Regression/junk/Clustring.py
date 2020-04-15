import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets
iris = datasets.load_iris()

df = pd.DataFrame({
        'x' : iris.data[:,0],
        'y' : iris.data[:,1],
        'cluster' : iris.target
        })
#centers
centroids= {}
for i in range(3):
        result_list = []
        result_list.append(df.loc[df['cluster'] == i]['x'].mean())
        result_list.append(df.loc[df['cluster'] == i]['y'].mean())
        
        centroids[i] = result_list
        
centroids
#plot
fig = plt.figure(figsize=(5, 5))
plt.scatter(df['x'], df['y'], c=iris.target)
plt.xlabel('Spea1 Length', fontsize=18)

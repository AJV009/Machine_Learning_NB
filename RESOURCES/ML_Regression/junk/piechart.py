import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
series=pd.Series(20 * np.random.rand(5),index =['a', 'b', 'c', 'd', 'e'], name ='series')
series.plot.pie(figsize =(4, 4))
plt.show
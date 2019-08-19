import numpy as np
import pandas as pd
from pandas import Series, DataFrame

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline

# Importing the ML library and loading already available dataset
from sklearn.datasets import load_boston

boston = load_boston()

# To see the attributes and target
print(boston.DESCR)

# Histogram of price vs number of houses
plt.hist(boston.target, bins = 50)
plt.xlabel("Prices in $1000's")
plt.ylabel('Number of houses')


plt.scatter(boston.data[:, 5], boston.target)
plt.ylabel("Price in $1000's")
plt.xlabel('Number of rooms')


# Trying to achieve the same objective using pandas and seaborn
boston_df = DataFrame(boston.data)
boston_df.columns = boston.feature_names
boston_df.head()
boston_df['Price'] = boston.target
sns.lmplot('RM', 'Price', data = boston_df)


# Using Numpy to find the best fit line for the dataset
X = boston_df.RM
X = np.vstack(boston_df.RM)
X = np.array([[value, 1] for value in X])

Y = boston_df.Price
m , b = np.linalg.lstsq(X,Y)[0]

# Creating a scatter plot
plt.plot(boston_df.RM, boston_df.Price, 'o')

# Plotting the best fit line from least square method
x = boston_df.RM
plt.plot(x, m*x+b, label = "Best Fit Line")


# Calculating the Root Mean Square Error (It is approximately equal to standard deviation) 
result = np.linalg.lstsq(X,Y)

error_total = result[1]

rmse = np.sqrt(error_total/len(X))
print('The root mean square error was %.2f' %rmse)




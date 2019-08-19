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



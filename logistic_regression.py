import numpy as np
import pandas as pd
import seaborn as sns
from pandas import Series, DataFrame
import math
import matplotlib.pyplot as plt
sns.set_style('Whitegrid')

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn import metrics
import statsmodels.api as sm

#Loading the dataset into a dataframe
df = sm.datasets.fair.load_pandas().data

def affair_check(x):
    if x != 0:
        return 1
    else:
        return 0

# Creating a new column to show if any amount of time was spent in affair
 df['Had_Affair'] = df['affairs'].apply(affair_check)

# Groupby 'Had_Affair' column
df.groupby('Had_Affair').mean()

# Now let's visualize the data
sns.catplot('age', data = df, kind = 'count', hue = 'Had_Affair')

sns.catplot('yrs_married', data = df, hue = 'Had_Affair', kind = 'count')

sns.catplot('children', data = df, hue = 'Had_Affair', kind = 'count')

sns.catplot('educ', data = df, hue = 'Had_Affair', kind = 'count')


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

# Data preparation for logistic regression
occ_dummies = pd.get_dummies(df['occupation'])
hus_occ_dummies = pd.get_dummies(df['occupation_husb'])

occ_dummies.columns = ['occ1', 'occ2', 'occ3', 'occ4', 'occ5', 'occ6']
occ_dummies.head()

hus_occ_dummies.columns = ['hocc1', 'hocc2', 'hocc3', 'hocc4', 'hocc5', 'hocc6']
hus_occ_dummies.head()

X = df.drop(['occupation', 'occupation_husb', 'Had_Affair'], axis = 1)

dummies = pd.concat([occ_dummies, hus_occ_dummies], axis = 1)
dummies.head()

X = pd.concat([X, dummies], axis = 1)

# Creating the target variable
Y = df.Had_Affair


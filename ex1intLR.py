import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
import itertools as it
from tqdm import tqdm
import pba
import tikzplotlib

from LRF import *

def generate_results(data):

    results = pd.Series(index = data.index, dtype = 'bool')
    for row in data.index:

        results[row] = sum(data.loc[row]) >= len(data.columns)*(15+5*np.random.randn())
    
    return results



# set seed to ensure same data
np.random.seed(10)

# Params
many = 50
dim = 3
some = 100
eps = 0.5

# Generate data
data = pd.DataFrame(40*np.random.rand(many,dim))
results = generate_results(data)

UQdata = pd.DataFrame({
    '1':[pba.I(data.iloc[i,0]-eps,data.iloc[i,0]+eps) for i in data.index],
    '2':[pba.I(data.iloc[i,1]-2*eps,data.iloc[i,1]+3*eps) for i in data.index],
    '3':[data.iloc[i,0] for i in data.index]
    }, dtype = 'O')

# Generate test data 
np.random.seed(111)
test_data = pd.DataFrame(30*np.random.rand(some,dim))
test_results = generate_results(test_data)

# Fit base model
base = LogisticRegression()
base.fit(data.to_numpy(),results.to_numpy())
# Classify test data
base_predict = base.predict(test_data)

# # fit int model
models = int_logistic_regression(UQdata,results)
print(models)

# for u,r in zip(UQdata,results.to_list()):
#     plt.plot([u.Left,u.Right],[r,r], marker='|')
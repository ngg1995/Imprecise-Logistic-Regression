#%%
from multiprocessing import freeze_support
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# from sklearn.linear_model import LogisticRegression
import itertools as it
from tqdm import tqdm
import pba
import random
import tikzplotlib
import matplotlib

import pymc3 as pm
import theano
import theano.tensor as tt
import arviz as az 

col_precise = 'black'
col_points = '#A69888'
col_ilr = '#4169E1'
col_ilr2 = '#5d2e46'
col_ilr3 = '#FF8C00'
col_ilr4 = '#008000'
col_mid = '#DC143C'

from LRF import *
from ImpLogReg import *

def intervalise(val,eps,method,bias=0,bounds = None):
   
    if method == 'u':
        m = np.random.uniform(val-eps,val+eps)
    elif method == 'b':
        m = val - eps + (1-bias)*eps
    elif method == 't':
        m = np.random.triangular(val-eps, val - eps + 2*bias*eps,val+eps)
   
    if bounds is not None:
        if m-eps < bounds[0]:
            return pba.I(bounds[0],m+eps)
        elif m+eps >bounds[1]:
            return pba.I(m-eps,bounds[1])
       
    return pba.I(m-eps,m+eps)
def midpoints(data):
    n_data = data.copy()
    for c in data.columns:
        for i in data.index:
            if data.loc[i,c].__class__.__name__ == 'Interval':

                n_data.loc[i,c] = data.loc[i,c].midpoint()

            
    return n_data

def generate_results(data):
    # set seed for reproducability
    np.random.seed(10)
    results = pd.Series(index = data.index, dtype = 'bool')
    
    for i in data.index:

        results.loc[i] = data.loc[i,0] >= 5+2*np.random.randn()    
        
    return results


# %%
### Generate Data
# set seed for reproducability
s = 1234
np.random.seed(s)
random.seed(s)

# Params
some = 210 # training datapoints
many = 100 # many test samples

train_data = pd.DataFrame(10*np.random.rand(some,1))
train_results = generate_results(train_data)

test_data = pd.DataFrame(10*np.random.rand(many,1))
test_results = generate_results(test_data)

# Intervalise data
eps = 0.375

UQdata = pd.DataFrame({
    0:[intervalise(train_data.iloc[i,0],eps,'t',0.8,bounds = (0,10)) for i in train_data.index]
    }, dtype = 'O')

#%%
if __name__ == "__main__":
    freeze_support()
    x_0 = train_data
    x_c = train_data - train_data.mean()
    y_simple = train_results
    
    with pm.Model() as model_simple:
        α = pm.Normal('α', mu=0, sd=10)
        β = pm.Normal('β', mu=0, sd=10)
        
        μ = α + pm.math.dot(x_c, β)    
        print(μ)
        θ = pm.Deterministic('θ', pm.math.sigmoid(μ))
        bd = pm.Deterministic('bd', -α/β)
        
        y_1 = pm.Bernoulli('y_1', p=θ, observed=y_simple)

        trace_simple = pm.sample(1000, tune=1000)
        
    #%%
    print(trace_simple)
    theta = trace_simple['θ'].mean(axis=0)
    print(az.summary(trace_simple, var_names=['α', 'β']))

    # idx = np.argsort(x_c)
    # print(idx)
    plt.plot(x_c, theta, lw=3)
    # plt.show()
    # plt.vlines(trace_simple['bd'].mean(), 0, 1, color='k')
    # # bd_hpd = az.hpd(trace_simple['bd'])
    # # plt.fill_betweenx([0, 1], bd_hpd[0], bd_hpd[1], color='k', alpha=0.5)

    # plt.scatter(x_c, np.random.normal(y_simple, 0.02),marker='.')
    # # az.plot_hpd(x_c, trace_simple['θ'], color='C2')

    # plt.xlabel('x')
    # plt.ylabel('θ', rotation=0)
    # locs, _ = plt.xticks()
    # plt.xticks(locs, np.round(locs + x_0.mean(), 1))
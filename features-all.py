# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
import itertools as it
from tqdm import tqdm
import pba
import random
import tikzplotlib
import matplotlib

font = {'size'   : 14,'family' : 'Times New Roman'}
matplotlib.rc('font', **font)
plt.rcParams['text.usetex'] = True

from ImpLogReg import *
from LRF import *


# colors
col_precise = 'black'
col_points = '#A69888'
col_ilr = '#4169E1'
col_ilr2 = '#5d2e46'
col_ilr3 = '#FF8C00'
col_ilr4 = '#008000'
col_mid = '#DC143C'

def intervalise(val,eps,method,b=0.5,bounds = None):
    np.random.seed(100)
    if method == 'u':
        m = np.random.uniform(val-eps,val+eps)
    elif method == 'b':
        m = val - eps + 2*b*eps
    elif method == 't':

        m = np.random.triangular(val-eps,val+b*eps,val+eps)
    
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

def get_sample(data,r = None):
    
    n_data = data.copy()
    
    for c in data.columns:
        
        for i in data.index:
            
            if data.loc[i,c].__class__.__name__ == 'Interval':
                
                if r is not None:
                    n_data.loc[i,c] = data.loc[i,c].left + r*data.loc[i,c].width()
                else:
                    n_data.loc[i,c] = data.loc[i,c].left + np.random.random()*data.loc[i,c].width()
            
    return n_data
# %%
### Generate Data
# set seed for reproducability
s = 1234
np.random.seed(s)
random.seed(s)

# Params
some = 50 # training datapoints
many = 100 # many test samples

train_data = pd.DataFrame(10*np.random.rand(some,1))
train_results = generate_results(train_data)

test_data = pd.DataFrame(10*np.random.rand(many,1))
test_results = generate_results(test_data)

#%%
### Fit logistic regression model
base = LogisticRegression()
base.fit(train_data.to_numpy(),train_results.to_numpy())

# Intervalise data
eps = 0.5

UQdata = pd.DataFrame({
    0:[intervalise(train_data.iloc[i,0],eps,'t',0.5,bounds = (0,10)) for i in train_data.index]
    }, dtype = 'O')


#%%
### Fit UQ models
ilr = ImpLogReg(uncertain_data=True, max_iter = 1000)
ilr.fit(UQdata,train_results,False)

#%%
fig, ax = plt.subplots()
many = 10
steps = 300
lX = np.linspace(0,10,steps)

for i in range(many+10):
    if i < 10:
        n_data = get_sample(UQdata,r = i/10)
    else:
        n_data = get_sample(UQdata)
    
    lr = LogisticRegression()
    lr.fit(n_data, train_results)
    
    lY = lr.predict_proba(lX.reshape(-1, 1))[:,1]
    
    ax.plot(lX,lY, color='grey', linewidth = 1)
    

for m,c,l in zip(ilr,[col_ilr,col_ilr2,col_ilr3,col_ilr4,col_mid,col_precise],[r"$\underline{E}$",r"$\underline{E}$",r"$E^\prime_{\underline{\beta_0}}$",r"$E^\prime_{\underline{\beta_1}}$",r"$E^\prime_{\overline{\beta_0}}$",r"$E^\prime_{\overline{\beta_1}}$"]):
    lY = m.predict_proba(lX.reshape(-1, 1))[:,1]
    ax.plot(lX,lY, color=c, linewidth = 2,label = l)
ax.legend()

for u,m,r in zip(UQdata[0],train_data[0],train_results.to_list()):
    yd = np.random.uniform(0,0.1)
    # plt.plot(m,r+yd,color = 'b',marker = 'x')
    if r == 0:
        yd = -yd
    ax.plot([u.left,u.right],[r+yd,r+yd],color = col_points, marker='|')
# %%
# fig.show()
tikzplotlib.save('figs/features-all.tikz',figure = fig,externalize_tables = True, override_externals = True,tex_relative_path_to_data = 'dat/')

# %%
# %%

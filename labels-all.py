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
# import matplotlib
# font = {'size'   : 14,'family' : 'Times New Roman'}
# matplotlib.rc('font', **font)
plt.rcParams['text.usetex'] = True

# colors
col_precise = 'black'
col_points = '#A69888'
col_ilr = '#4169E1'
col_ilr2 = '#5d2e46'
col_ilr3 = '#FF8C00'
col_ilr4 = '#008000'
col_mid = '#DC143C'

from ImpLogReg import *
from LRF import *


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



# %%
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



# %%
### Intervalise data
eps = 0.375

# drop some results
few = 5 #uncertain points
random.seed(12345) # for reproducability
uq_data_index = random.sample([i for i in train_data.index if abs(train_data.loc[i,0]-5) <= 1.5], k = few) # clustered around center

uq_data = train_data.loc[uq_data_index]
uq_results = pd.Series([int(train_results.loc[i]) if i not in uq_data_index else pba.I(0,1) for i in train_results.index], index = train_data.index, dtype='O')
nuq_data = train_data.loc[[i for i in train_data.index if i not in uq_data_index]]
nuq_results = train_results.loc[[i for i in train_data.index if i not in uq_data_index]]

# %% [markdown]
### Fit UQ models
ilr = ImpLogReg(uncertain_class=True, max_iter = 1000)
ilr.fit(train_data,uq_results)

# %% 
### Plot results
steps = 300
lX = np.linspace(0,10,steps)

fig,ax = plt.subplots()

for i in it.product([False,True],repeat = few):
    print(i)
    
    new_data = pd.concat((nuq_data,train_data.loc[[i for i in train_data.index if i in uq_data_index]]), ignore_index = True)
    new_results = pd.concat((nuq_results, pd.Series(i)), ignore_index = True)
    
    lr = LogisticRegression()
    lr.fit(new_data.to_numpy(),new_results)
    lY = lr.predict_proba(lX.reshape(-1, 1))[:,1]
    ax.plot(lX,lY, color='grey', linewidth = 0.5)


ax.set_xlabel('$x$')
ax.set_ylabel('$\pi(x)$')
ax.scatter(nuq_data,nuq_results,color=col_points,zorder=10)
for i in uq_data_index:

    ax.plot([uq_data.loc[i],uq_data.loc[i]],[0,1],color=col_points)


for m,c,l in zip(ilr,[col_ilr,col_ilr2,col_ilr3,col_ilr4,col_mid,col_precise],[r"$\underline{E}$",r"$\underline{E}$",r"$E^\prime_{\underline{\beta_0}}$",r"$E^\prime_{\underline{\beta_1}}$",r"$E^\prime_{\overline{\beta_0}}$",r"$E^\prime_{\overline{\beta_1}}$"]):
    lY = m.predict_proba(lX.reshape(-1, 1))[:,1]
    ax.plot(lX,lY, color=c, linewidth = 2,label = l)
ax.legend()
#%%
tikzplotlib.save('figs/labels-all.tikz',figure = fig,externalize_tables = True, override_externals = True,tex_relative_path_to_data = 'dat/')#%%

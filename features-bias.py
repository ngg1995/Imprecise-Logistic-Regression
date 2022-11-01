#%%
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

col_precise = 'black'
col_points = '#A69888'
col_ilr = '#4169E1'
col_ilr2 = '#5d2e46'
col_ilr3 = '#FF8C00'
col_ilr4 = '#008000'
col_mid = '#DC143C'

from LRF import *
# from ImpLogReg import *
from old import ImpLogReg
from other_methods import *


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
#Load dataset 
from dataset import train_data, train_results, test_data, test_results

#%%
### Fit logistic regression model
base = LogisticRegression()
base.fit(train_data.to_numpy(),train_results.to_numpy())

# Intervalise data
eps = 1

def low_val(data):
    return pd.DataFrame({
    0:[intervalise(data.iloc[i,0],eps,'b',-1,(0,10)) for i in data.index]
    }, dtype = 'O')
   
def hi_val(data):
    return pd.DataFrame({
    0:[intervalise(data.iloc[i,0],eps,'b',1,(0,10)) for i in data.index]
    }, dtype = 'O') 

def ex3(data):
    n_data = []
    for i in data.index:
        if data.iloc[i,0] < 2.5:
            n_data.append(
                data.iloc[i,0]
            )
        elif data.iloc[i,0] < 5:
            n_data.append(
                intervalise(data.iloc[i,0],0.25,'u',-1,(0,10))
            )
        elif data.iloc[i,0] < 8: 
            n_data.append(
                intervalise(data.iloc[i,0],0.75,'u',-1,(0,10))
            )
        else:
            n_data.append(
                intervalise(data.iloc[i,0],0.75,'b',-1,(0,10))
            )
    return pd.DataFrame({0:n_data}, dtype = 'O')
       
def ex4(data,results):
    n_data = []
    for i in data.index:
        if results.loc[i]:
            n_data.append(
                intervalise(data.iloc[i,0],0.5,'b',1,(0,10))
            )
        else:
            n_data.append(
                intervalise(data.iloc[i,0],0.5,'b',-1,(0,10))
            )
    return pd.DataFrame({0:n_data}, dtype = 'O')
        
UQdatasets = [
    low_val(train_data),
    hi_val(train_data),
    ex3(train_data),
    ex4(train_data,train_results)
]


#%%
fig, axs = plt.subplots(2,2,sharey = True)

for jj, UQdata,ax in zip([0,1,2,3],UQdatasets,np.ravel(axs)):

    ### Fit models with midpoint data
    mid_data = midpoints(UQdata)
    mid = LogisticRegression(max_iter=1000)
    mid.fit(mid_data.to_numpy(),train_results.to_numpy())

    ### Fit UQ models
    ilr = ImpLogReg(uncertain_data=True, max_iter = 1000)
    ilr.fit(UQdata,train_results)
    
    ### Fit UQ models
    ilr_fast = ImpLogReg(uncertain_data=True, max_iter = 1000)
    ilr_fast.fit(UQdata,train_results,fast=True,n_p_vals = 100)    
    
    ### Fit de Souza model
    ds = DSLR(max_iter = 1000)
    ds.fit(mid_data,train_results)
    
    ### Fit Billard--Diday model
    bd = BDLR(max_iter = 1000)
    bd.fit(UQdata,train_results,N = 10000)
    
    ### Plot results
    steps = 300
    lX = np.linspace(0,10,steps)
    lY = base.predict_proba(lX.reshape(-1, 1))[:,1]
    lYn = mid.predict_proba(lX.reshape(-1, 1))[:,1]
    lYu = ilr.predict_proba(lX.reshape(-1,1))[:,1]
    lYf = ilr_fast.predict_proba(lX.reshape(-1,1))[:,1]
    lYd = ds.predict_proba(lX.reshape(-1,1))[:,1]
    lYb = bd.predict_proba(lX.reshape(-1,1))[:,1]
    # plt.set_xlabel('$x$')
    # plt.set_ylabel('$\pi(x)$')

    ax.plot(lX,lY,color=col_precise,zorder=10,lw=2,label = 'base')
    ax.plot(lX,lYn,color=col_mid,zorder=10,lw=2,label = 'mid')
    ax.plot(lX,lYd,color=col_ilr2,zorder=10,lw=3,label = 'ds',linestyle = '--')
    ax.plot(lX,lYb,color=col_ilr3,zorder=10,lw=4,label = 'bd',linestyle = ':')

    for u,m,r in zip(UQdata[0],train_data[0],train_results.to_list()):
        yd = np.random.uniform(-0.0,0.1)
        if r == 0: yd = -yd
        # plt.plot(m,r+yd,color = 'b',marker = 'x')
        if not isinstance(u,pba.Interval): u = pba.I(u)
        ax.plot([u.left,u.right],[r+yd,r+yd],color = col_points, marker='|')
        
    ax.plot(lX,[i.left for i in lYu],color=col_ilr,lw=2)
    ax.plot(lX,[i.right for i in lYu],color=col_ilr,lw=2,label = 'ALG3')
    
    ax.plot(lX,[i.left for i in lYu],color=col_ilr,lw=2)
    ax.plot(lX,[i.right for i in lYu],color=col_ilr,lw=2,label = 'ALG4')
    
    nmin = np.full(len(lX),np.inf)
    nmax = np.full(len(lX),-np.inf)
    
    for lr in bd:
        lY = lr.predict_proba(lX.reshape(-1, 1))[:,1]
        
        nmin = np.minimum(nmin.ravel(),lY.ravel())
        nmax = np.maximum(nmax.ravel(),lY.ravel())
        
    ax.plot(lX,nmin,'r')
    ax.plot(lX,nmax,'r',label = 'MC')
    # ax.legend()
    # plt.show()

#%%
# fig.savefig('../LR-paper/figs/biased_int.png',dpi = 600)
# fig.savefig('figs/biased_int.png',dpi = 600)

tikzplotlib.save("figs/biased_int.tikz",figure = fig,externalize_tables = True, override_externals = True,tex_relative_path_to_data = 'dat/')


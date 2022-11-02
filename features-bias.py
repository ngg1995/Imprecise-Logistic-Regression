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
from ImpLogReg import *
# from old import ImpLogReg
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
                pba.I(data.iloc[i,0])
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
# fig, axs = plt.subplots(2,2,sharey = Tr
for jj, UQdata in zip([0,1,2,3],UQdatasets):

    ### Fit models with midpoint data
    mid_data = midpoints(UQdata)
    mid = LogisticRegression(max_iter=1000)
    mid.fit(mid_data.to_numpy(),train_results.to_numpy())

    ### Fit UQ models
    ilr = ImpLogReg(uncertain_data=True, max_iter = 1000)
    ilr.fit(UQdata,train_results)
    
    ### Fit UQ models
    fast = ImpLogReg(uncertain_data=True, max_iter = 1000)
    fast.fit(UQdata,train_results,fast=True,n_p_vals = 100)    
    
    ### Fit de Souza model
    ds = DSLR(max_iter = 1000)
    ds.fit(mid_data,train_results)
    
    ### Fit Billard--Diday model
    bd = BDLR(max_iter = 1000)
    bd.fit(UQdata,train_results,N = 10000)

    def make_dat_file(fname,x,y):
        with open(f'figs/dat/features/{fname}.dat','w') as f:
            for i,j in zip(x,y):
                print(f"{i} {j}",file = f)

    steps = 300
    lX = np.linspace(0,10,steps)

    lY = base.predict_proba(lX.reshape(-1, 1))[:,1]
    make_dat_file(f'bias-{jj}-base',lX,lY)

    lYm = mid.predict_proba(lX.reshape(-1, 1))[:,1]
    make_dat_file(f'bias-{jj}-mid',lX,lYm)

    lYd = ds.predict_proba(lX.reshape(-1,1))[:,1]
    make_dat_file(f'bias-{jj}-deSouza',lX,lYd)

    lYb = bd.predict_proba(lX.reshape(-1,1))[:,1]
    make_dat_file(f'bias-{jj}-Billard-Diday',lX,lYb)

    lYu = ilr.predict_proba(lX.reshape(-1,1))[:,1]
    make_dat_file(f'bias-{jj}-minmaxcoef-right',lX,[i.right for i in lYu])
    make_dat_file(f'bias-{jj}-minmaxcoef-left',lX,[i.left for i in lYu])

    mc_min = np.full(len(lX),np.inf)
    mc_max = np.full(len(lX),-np.inf)
    for lr in bd:
        lY = lr.predict_proba(lX.reshape(-1, 1))[:,1]
        mc_min = np.minimum(mc_min.ravel(),lY.ravel())
        mc_max = np.maximum(mc_max.ravel(),lY.ravel())
    make_dat_file(f'bias-{jj}-MC-right',lX,mc_min)
    make_dat_file(f'bias-{jj}-MC-left',lX,mc_max)

    lYu = fast.predict_proba(lX.reshape(-1,1))[:,1]
    make_dat_file(f'bias-{jj}-fast-right',lX,[i.right for i in lYu])
    make_dat_file(f'bias-{jj}-fast-left',lX,[i.left for i in lYu])

    with open(f"figs/dat/features/bias-{jj}-intervals",'w') as f:
        jit = np.random.default_rng(1)
        for iii in UQdata.index:
            if train_results.loc[iii]:
                jitter = jit.uniform(0,0.1)
            else:
                jitter = jit.uniform(-0.1,0)
                
            print(f"\\addplot [semithick, points, mark=|, mark size=3, mark options={{solid}}, forget plot]\ntable {{% \n{UQdata.loc[iii,0].left} {train_results.loc[iii] + jitter}\n{UQdata.loc[iii,0].right} {train_results.loc[iii] + jitter}\n}};",file = f)
            

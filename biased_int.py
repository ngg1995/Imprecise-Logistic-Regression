import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
import itertools as it
from tqdm import tqdm
import pba
import random

import matplotlib
font = {'size'   : 14}
matplotlib.rc('font', **font)

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


### Generate Data
# set seed for reproducability
np.random.seed(1)
random.seed(2)

# Params
some = 50 # training datapoints
many = 500 # many test samples

train_data = pd.DataFrame(10*np.random.rand(some,1))
train_results = generate_results(train_data)

test_data = pd.DataFrame(10*np.random.rand(many,1))
test_results = generate_results(test_data)

### Fit logistic regression model
base = LogisticRegression()
base.fit(train_data.to_numpy(),train_results.to_numpy())

# Intervalise data
eps = 0.25

UQdatasets = [
    pd.DataFrame({
    0:[intervalise(train_data.iloc[i,0],eps,'b',-1,(0,10)) for i in train_data.index]
    }, dtype = 'O'),
    pd.DataFrame({
    0:[intervalise(train_data.iloc[i,0],eps,'b',1,(0,10)) for i in train_data.index]
    }, dtype = 'O'),
    pd.DataFrame({
    0:[intervalise(train_data.iloc[i,0],eps,'b',-1,(0,10)) if train_data.iloc[i,0] > 5 else intervalise(train_data.iloc[i,0],eps,'b',1,(0,10)) for i in train_data.index]
    }, dtype = 'O')
    ]

for jj, UQdata in zip([0,1,2],UQdatasets):

    ### Fit logistic regression model
    base = LogisticRegression()
    base.fit(train_data.to_numpy(),train_results.to_numpy())

    # Intervalise data
    eps = 0.25
    ### Fit logistic regression model on full dataset
    base = LogisticRegression(max_iter=1000)
    base.fit(train_data.to_numpy(),train_results.to_numpy())


    ### Fit models with midpoint data
    nuq_data = midpoints(UQdata)
    nuq = LogisticRegression(max_iter=1000)
    nuq.fit(nuq_data.to_numpy(),train_results.to_numpy())

    ### Fit UQ models
    ilr = ImpLogReg(uncertain_data=True, max_iter = 1000)
    ilr.fit(UQdata,train_results)

    ### Plot results
    steps = 300
    lX = np.linspace(0,10,steps)
    lY = base.predict_proba(lX.reshape(-1, 1))[:,1]
    lYn = nuq.predict_proba(lX.reshape(-1, 1))[:,1]
    lYu = ilr.predict_proba(lX.reshape(-1,1))[:,1]

    plt.xlabel('$x$')
    plt.ylabel('$\pi(x)$')

    plt.plot(lX,lY,color='k',zorder=10,lw=2,label = 'Truth')
    plt.plot(lX,lYn,color='#DC143C',zorder=10,lw=2,label = 'No UQ')

    for u,m,r in zip(UQdata[0],train_data[0],train_results.to_list()):
        yd = np.random.uniform(-0.05,0.05)
        # plt.plot(m,r+yd,color = 'b',marker = 'x')
        plt.plot([u.left,u.right],[r+yd,r+yd],color = 'grey', marker='|')
        
    plt.plot(lX,[i.left for i in lYu],color='#4169E1',lw=2)
    plt.plot(lX,[i.right for i in lYu],color='#4169E1',lw=2,label = 'Uncertainty Bounds')

    plt.savefig('../LR-paper/figs/biased_int_%i.png'%jj,dpi = 600)
    plt.savefig('figs/biased_int_%i.png'%jj,dpi = 600)

    plt.clf()


    ### Hosmer-Lemeshow
    hl_b, pval_b = hosmer_lemeshow_test(base,train_data,train_results,g = 10)

    hl_nuq, pval_nuq = hosmer_lemeshow_test(nuq,train_data,train_results,g = 10)
    #
    hl_uq, pval_uq = UQ_hosmer_lemeshow_test(ilr,train_data,train_results,g = 10)

    with open('runinfo/biased_int_HL_%i.out'%jj,'w') as f:
        print('base\nhl = %.3f, p = %.3f' %(hl_b,pval_b),file = f)
        print('no UQ\nhl = %.3f, p = %.3f' %(hl_nuq,pval_nuq),file = f) 

        print('UQ\nhl = [%.3f,%.3f], p = [%.3f,%.3f]' %(*hl_uq,*pval_uq),file = f) 
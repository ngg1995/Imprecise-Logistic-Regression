# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
import itertools as it
import tikzplotlib
from tqdm import tqdm
import pba
import random
from sklearn.semi_supervised import SelfTrainingClassifier
import matplotlib
import tikzplotlib
# font = {'size'   : 14,'family' : 'Times New Roman'}
# matplotlib.rc('font', **font)

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

from dataset import *

# %% [markdown]
### Fit logistic regression model on full dataset
base = LogisticRegression(max_iter=1000)
base.fit(train_data.to_numpy(),train_results.to_numpy())

#%%
def random_5(data,results):
    few = 5
    random.seed(1)
    uq_data_index = random.sample([i for i in data.index], k = few)
    return uq_data_index
       
def random_20(data,results):
    few = 20
    random.seed(2)
    uq_data_index = random.sample([i for i in data.index], k = few)
    return uq_data_index
    
def hi_10(data,results):
    few = 5 #uncertain points
    random.seed(3) # for reproducability
    uq_data_index = random.sample([i for i in data.index if train_data.loc[i,0] > 6], k = few) # clustered around center
    return uq_data_index

def lo_20(data,results):
    few = 15 #uncertain points
    random.seed(4) # for reproducability
    uq_data_index = random.sample([i for i in data.index if train_data.loc[i,0] < 4], k = few) # clustered around center
    return uq_data_index
        
def ones_8(data,results):
    few = 8 #uncertain points
    random.seed(1) # for reproducability
    uq_data_index = random.sample([i for i in results[results].index], k = few) # clustered around center
    return uq_data_index

def zeroes_8(data,results):
    few = 8 #uncertain points
    random.seed(6) # for reproducability
    uq_data_index = random.sample([i for i in results[~results].index], k = few) # clustered around center
    return uq_data_index
        
UQdatasets = [
    random_5(train_data,train_results),
    random_20(train_data,train_results),
    lo_20(train_data,train_results),
    hi_10(train_data,train_results)
    # ones_8(train_data,train_results),
    # zeroes_8(train_data,train_results)
]

# %% [markdown]
fig, axs = plt.subplots(2,2,sharey = True)

for jj, uq_data_index ,ax in zip(range(len(UQdatasets)),UQdatasets,np.ravel(axs)):
    uq_data = train_data.loc[uq_data_index]
    uq_results = pd.Series([int(train_results.loc[i]) if i not in uq_data_index else pba.I(0,1) for i in train_results.index], index = train_data.index, dtype='O')

    ### Fit models with no uq data
    nuq_data = train_data.loc[[i for i in train_data.index if i not in uq_data_index]]
    nuq_results = train_results.loc[[i for i in train_data.index if i not in uq_data_index]]
    nuq = LogisticRegression(max_iter=1000)
    nuq.fit(nuq_data.to_numpy(),nuq_results.to_numpy())

    sslr_results = pd.Series([int(train_results.loc[i]) if i not in uq_data_index else -1 for i in train_results.index], index = train_data.index, dtype=int)
    ### Fit Semi-supervised logsitic regression
    sslr = SelfTrainingClassifier(LogisticRegression())
    sslr.fit(train_data,sslr_results)

    ### Fit UQ models
    ilr = ImpLogReg(uncertain_class=True, max_iter = 1000)
    ilr.fit(train_data,uq_results)
    
    ### Plot results
    steps = 300
    lX = np.linspace(0,10,steps)
    lY = base.predict_proba(lX.reshape(-1, 1))[:,1]
    lYn = nuq.predict_proba(lX.reshape(-1, 1))[:,1]
    lYs = sslr.predict_proba(lX.reshape(-1, 1))[:,1]
    lYu = ilr.predict_proba(lX.reshape(-1,1))[:,1]

    ax.plot(lX,lY,color=col_precise,zorder=10,lw=2,label = '$\mathcal{LR}(D)$') 
    ax.plot(lX,lYn,color=col_mid,zorder=10,lw=2,label = '$\mathcal{LR}(F_\\times)$') 
    ax.plot(lX,lYs,color=col_ilr4,zorder=10,lw=2,label = r'$ss$') 
    ax.scatter(nuq_data,nuq_results,color=col_points,zorder=10)
    for i in uq_data_index:

        ax.plot([uq_data.loc[i],uq_data.loc[i]],[0,1],color=col_points)
        # plt.scatter(uq_data.loc[i],train_results.loc[i],marker = 'd',color = 'grey',zorder = 14)

    ax.plot(lX,[i.left for i in lYu],color=col_ilr,lw=2)
    ax.plot(lX,[i.right for i in lYu],color=col_ilr,lw=2,label = '$\mathcal{ILR}(F)$')
# %%
tikzplotlib.save("figs/biased_labels.tikz",figure = fig,externalize_tables = True, override_externals = True,tex_relative_path_to_data = 'dat/')

# %%

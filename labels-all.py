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
plt.rcParams['text.usetex'] = True
# import matplotlib
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

# %%
### Intervalise data
eps = 0.375

# drop some results
few = 5 #uncertain points
random.seed(12345) # for reproducability
uq_data_index = random.sample([i for i in train_data.index ], k = few) # clustered around center

uq_data = train_data.loc[uq_data_index]
uq_results = pd.Series([int(train_results.loc[i]) if i not in uq_data_index else pba.I(0,1) for i in train_results.index], index = train_data.index, dtype='O')
nuq_data = train_data.loc[[i for i in train_data.index if i not in uq_data_index]]
nuq_results = train_results.loc[[i for i in train_data.index if i not in uq_data_index]]


#%% 
### Fit UQ models
ilr = ImpLogReg(uncertain_class=True, max_iter = 1000)
ilr.fit(train_data,uq_results,simple=True)

all_models = []
new_results = train_results.copy()
for i in it.product([False,True],repeat=few):
    new_results.loc[uq_data_index] = i
    all_models.append(
        LogisticRegression().fit(train_data,new_results)
    )
# %% 
### Plot results
steps = 300
lX = np.linspace(0,10,steps)

lYu = ilr.predict_proba(lX.reshape(-1,1))[:,1]

fig1, ax1 = plt.subplots()

ax1.set_xlabel('$x$')
ax1.set_ylabel('$\pi(x)$')

ax1.scatter(nuq_data,nuq_results,color=col_points,zorder=10)

for m in all_models:
    lY = m.predict_proba(lX.reshape(-1,1))[:,1]
    ax1.plot(lX,lY, color='grey', linewidth = 1)
    
for i in uq_data_index:
    ax1.plot([uq_data.loc[i],uq_data.loc[i]],[0,1],color=col_points)
    # plt.scatter(uq_data.loc[i],train_results.loc[i],marker = 'd',color = 'grey',zorder = 14)



for m,c,l in zip(ilr,[col_ilr,col_ilr2,col_ilr3,col_ilr4,col_mid,col_precise],[r"$\underline{E}$",r"$\underline{E}$",r"$E^\prime_{\underline{\beta_0}}$",r"$E^\prime_{\underline{\beta_1}}$",r"$E^\prime_{\overline{\beta_0}}$",r"$E^\prime_{\overline{\beta_1}}$"]):
    lY = m.predict_proba(lX.reshape(-1, 1))[:,1]
    ax1.plot(lX,lY, color=c, linewidth = 2,label = l)
ax1.legend()
ax1.plot(lX,[i.left for i in lYu],color=col_ilr,lw=2,linestyle = '--',)
ax1.plot(lX,[i.right for i in lYu],color=col_ilr,lw=2,linestyle = '--',label = '$\mathcal{ILR}(F)$')

#%%
tikzplotlib.save('figs/labels-all.tikz',figure = fig1,externalize_tables = True, override_externals = True,tex_relative_path_to_data = 'dat/')
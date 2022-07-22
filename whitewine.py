#%%
from scipy import rand
import tikzplotlib
from ImpLogReg import ImpLogReg
import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import itertools as it
from tqdm import tqdm
import pba

import matplotlib
font = {'size'   : 16,'family' : 'Times New Roman'}
matplotlib.rc('font', **font)

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

def deintervalise(data, binary_cols):
    n_data = data.copy()
    for c in data.columns:
        if c in binary_cols:
            continue
        for i in data.index:
            if data.loc[i,c].__class__.__name__ == 'Interval':
                n_data.loc[i,c] = data.loc[i,c].left + np.random.rand()*data.loc[i,c].width()
            
    return n_data

#%%
### Load whitewine dataset
whitewine = pd.read_csv('whitewine.csv',index_col = None)

X = whitewine[[c for c in whitewine.columns if c != 'quality']]
Y = whitewine.quality > 6

### Split into test and train samples
train_data, test_data, train_results, test_results = train_test_split(X, Y, test_size=0.75,random_state = 0)
print(len(train_data))

# %%
### Fit LR model on dataset
base = LogisticRegression(max_iter=1000)
base.fit(train_data.to_numpy(),train_results.to_numpy())

train_data_index = list(train_data.index)
uq_results_index = random.sample([i for i in train_data_index if (whitewine.loc[i,"quality"] == 6 or whitewine.loc[i,"quality"] == 7)], k = 20)
uq_results = pd.Series([pba.I(0,1) if i in uq_results_index else train_results.loc[i] for i in train_data_index],index = train_data_index, dtype='O')

#%%


#%%
### Fit ILR model
ilr = ImpLogReg(uncertain_data=0, uncertain_class=True, max_iter = 1000)
ilr.fit(train_data,uq_results)

# %% [markdown]
s,fpr,probabilities = ROC(model = base, data = test_data, results = test_results)
s_t, fpr_t, Sigma, Tau = incert_ROC(ilr, test_data, test_results)

s_i, fpr_i,ilr_probabilities = ROC(ilr, test_data, test_results)


xl = []
xu = []
yl = []
yu = []
for i,j in zip(fpr_i,s_i):
    
    if not isinstance(i,pba.Interval):
        i = pba.I(i)
    if not isinstance(j,pba.Interval):
        j = pba.I(j)
      
    xl.append(i.left)
    xu.append(i.right)
    
    yl.append(j.left)
    yu.append(j.right)
#%%
rocfig,axroc = plt.subplots(1,1)
axroc.plot(xl,yu, col_ilr,label = 'Imprecise Model')
axroc.plot(xu,yl, col_ilr )
axroc.plot([0,1],[0,1],linestyle = '--',color=col_points)
axroc.set(xlabel = '$fpr$',ylabel='$s$')
axroc.plot(fpr,s,color=col_precise,label = 'Base')
# axroc.plot(nuq_fpr,nuq_s,color=col_mid,linestyle = '--',label='Deintervalised')
# axroc.plot(fpr_t,s_t,col_ilr3,label='No Pwhiteiction')
axroc.legend()
# rocfig.savefig('figs/whitewine_ROC.eps',dpi = 600)
# rocfig.savefig('../SUM/whitewine_ROC.eps',dpi = 600)

tikzplotlib.savefig('figs/whitewine_roc.tikz',figure = rocfig,externalize_tables = True, override_externals = True,tex_relative_path_to_data = 'dat/')

#%%
with open('runinfo/whitewine_auc.out','w') as f:
    print('NO UNCERTAINTY: %.4f' %auc(s,fpr), file = f)
    # print('NO Pwhite: %.4f' %auc(s_t,fpr_t), file = f)
    print('ILR: [%.3f,%.3f]'  %(auc(yl,xu),auc(yu,xl)), file = f)

    
# %%
### Get confusion matrix
# Classify test data
c = 0.18
base_pwhiteict = [p>=c for p in base.pwhiteict_proba(test_data.to_numpy())[:,1]]

# CLASSIFY NO_UQ MODEL DATA 
# nuq_pwhiteict = [p>=c for p in nuq.pwhiteict_proba(test_data.to_numpy())[:,1]]

# CLASSIFY UQ MODEL 
ilr_pwhiteict = [p>=c for p in ilr.pwhiteict_proba(test_data.to_numpy())[:,1]]

with open('runinfo/whitewine_cm.out','w') as f:
    print('TRUE MODEL',file = f)
    a,b,c,d = generate_confusion_matrix(test_results,base_pwhiteict)
    print('TP=%i\tFP=%i\nFN=%i\tTN=%i' %(a,b,c,d),file = f)

    # Calculate sensitivity and specificity
    print('Sensitivity = %.3f' %(a/(a+c)),file = f)
    print('Specificity = %.3f' %(d/(b+d)),file = f)

    
    print('UQ MODEL',file = f)
    
    aaai,bbbi,ccci,dddi = generate_confusion_matrix(test_results,ilr_pwhiteict,throw = False)
    try:
        sssi = aaai/(a+c)
    except:
        sssi = None
    try:    
        ttti = dddi/(b+d)
    except:
        ttti = None

    print('TP=[%i,%i]\tFP=[%i,%i]\nFN=[%i,%i]\tTN=[%i,%i]' %(*aaai,*bbbi,*ccci,*dddi),file = f)

    # Calculate sensitivity and specificity
    print('Sensitivity = [%.3f,%.3f]\nSpecificity = [%.3f,%.3f]' %(*sssi,*ttti),file = f)

    
    aaa,bbb,ccc,ddd,eee,fff = generate_confusion_matrix(test_results,ilr_pwhiteict,throw = True)
    try:
        sss = 1/(1+ccc/aaa)
    except:
        sss = None
    try:    
        ttt = 1/(1+bbb/ddd)
    except:
        ttt = None
        
    print('TP=%i\tFP=%i\nFN=%i\tTN=%i\nNP(+)=%i\tNP(-)=%i' %(aaa,bbb,ccc,ddd,eee,fff),file = f)

    # Calculate sensitivity and specificity
    print('Sensitivity = %.3f' %(sss),file = f)
    print('Specificity = %.3f' %(ttt),file = f)
    print('sigma = %.3f' %(eee/(aaa+ccc+eee)),file = f)
    print('tau = %.3f' %(fff/(bbb+ddd+fff)),file = f)
   
# %%

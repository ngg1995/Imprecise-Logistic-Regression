#%%
import tikzplotlib
from ImpLogReg import ImpLogReg
import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import SelfTrainingClassifier
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
### Load redwine dataset
redwine = pd.read_csv('whitewine.csv',index_col = None)

X = redwine[[c for c in redwine.columns if c != 'quality']]
Y = redwine['quality'] >= 6

### Split into test and train samples
train_data, test_data, train_results, test_results = train_test_split(X, Y, test_size=0.5,random_state = 0)
print(len(train_data),sum(train_results))

train_data_index = list(train_data.index)
uq_data_index = random.sample([i for i in train_data_index if (redwine.loc[i,"quality"] == 6)], k = 40)
uq_results = pd.Series([pba.I(0,1) if i in uq_data_index else train_results.loc[i] for i in train_data_index],index = train_data_index, dtype='O')

#%%
### Fit logistic regression model on full dataset
base = LogisticRegression(max_iter=10000,solver='saga')
base.fit(train_data,train_results)

# %% [markdown]
### Fit models with no uq data
nuq_data = train_data.loc[[i for i in train_data.index if i not in uq_data_index]]
nuq_results = train_results.loc[[i for i in train_data.index if i not in uq_data_index]]
nuq = LogisticRegression(max_iter=10000,solver='saga')
nuq.fit(nuq_data,nuq_results)

#%%
### Fit Semi-Supervised Model
sslr_results = pd.Series([int(train_results.loc[i]) if i not in uq_data_index else -1 for i in train_results.index], index = train_data.index, dtype=int)

sslr = SelfTrainingClassifier(LogisticRegression(max_iter=10000,solver='saga'))
sslr.fit(train_data,sslr_results)

# %% [markdown]
### Fit UQ models
ilr = ImpLogReg(uncertain_class=True, max_iter = 10000,solver='saga')
ilr.fit(train_data,uq_results)

# %%
### Get confusion matrix
# Classify test data
base_predict = base.predict(test_data)

# CLASSIFY NO_UQ MODEL DATA 
nuq_predict = nuq.predict(test_data)

# CLASSIFY NO_UQ MODEL DATA 
sslr_predict = sslr.predict(test_data)

# CLASSIFY UQ MODEL 
ilr_predict = ilr.predict(test_data)

with open('runinfo/redwine_cm.out','w') as f:
    print('TRUE MODEL',file = f)
    a,b,c,d = generate_confusion_matrix(test_results,base_predict)
    print('TP=%i\tFP=%i\nFN=%i\tTN=%i' %(a,b,c,d),file = f)

    # Calculate sensitivity and specificity
    print('Sensitivity = %.3f' %(a/(a+c)),file = f)
    print('Specificity = %.3f' %(d/(b+d)),file = f)

    print('\nDISCARDED DATA MODEL',file = f)
    aa,bb,cc,dd = generate_confusion_matrix(test_results,nuq_predict)
    try:
        ss = 1/(1+cc/aa)
    except:
        ss = None
    try:    
        tt = 1/(1+bb/dd)
    except:
        tt = None
    print('TP=%s\tFP=%s\nFN=%s\tTN=%s' %(aa,bb,cc,dd),file = f)
    # Calculate sensitivity and specificity
    print('Sensitivity = %.3f' %(ss),file = f)
    print('Specificity = %.3f' %(tt),file = f)
    
    print('\nSSLR MODEL',file = f)
    aa,bb,cc,dd = generate_confusion_matrix(test_results,sslr_predict)
    try:
        ss = 1/(1+cc/aa)
    except:
        ss = None
    try:    
        tt = 1/(1+bb/dd)
    except:
        tt = None
    print('TP=%s\tFP=%s\nFN=%s\tTN=%s' %(aa,bb,cc,dd),file = f)

    # Calculate sensitivity and specificity
    print('Sensitivity = %.3f' %(ss),file = f)
    print('Specificity = %.3f' %(tt),file = f)
    
    print('\nUQ MODEL',file = f)
    
    aaai,bbbi,ccci,dddi = generate_confusion_matrix(test_results,ilr_predict,throw = False)
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
    print('Sensitivity = [%.3f,%.3f]\nSpecificity = [%.3f,%.3f]\n' %(*sssi,*ttti),file = f)

    
    aaa,bbb,ccc,ddd,eee,fff = generate_confusion_matrix(test_results,ilr_predict,throw = True)
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
   
# %% [markdown]
s,fpr,probabilities = ROC(model = base, data = test_data, results = test_results)
nuq_s,nuq_fpr,nuq_probabilities = ROC(model = nuq, data = test_data, results = test_results)
sslr_s,sslr_fpr,nuq_probabilities = ROC(model = sslr, data = test_data, results = test_results)
s_t, fpr_t, Sigma, Tau = incert_ROC(ilr, test_data, test_results)

s_i, fpr_i,ilr_probabilities = ROC(ilr, test_data, test_results)

densfig,axdens = plt.subplots(nrows = 2, sharex= True)

dat1 = ['x y']
dat2 = ['x y']
dat3 = ['x y']
dat4 = ['x y']

for i,(p,u,nuqp,r) in enumerate(zip(probabilities,ilr_probabilities,nuq_probabilities,test_results.to_list())):
    yd = np.random.uniform(-0.1,0.1)
    if r:
        dat1 += [f"{p} {yd}"]
        dat2 += [f"{nuqp} {0.21+yd}"]
        # axdens[0].scatter(p,yd,color = 'k',marker = 'o',alpha = 0.5)
        # axdens[0].scatter(nuqp,0.21+yd,color = col_mid,marker = 'o',alpha = 0.5)
        axdens[0].plot([*u],[yd-0.21,yd-0.21],color = col_ilr, alpha = 0.3)
        axdens[0].scatter([*u],[yd-0.21,yd-0.21],color = col_ilr, marker = '|')
    else:
        dat3 += [f"{p} {yd}"]
        dat4 += [f"{nuqp} {0.21+yd}"]
        # axdens[1].scatter(p,yd,color = 'k',marker = 'o',alpha = 0.5)
        # axdens[1].scatter(nuqp,0.21+yd,color = col_mid,marker = 'o',alpha = 0.5)
        axdens[1].plot([*u],[yd-0.21,yd-0.21],color = col_ilr, alpha = 0.3)
        axdens[1].scatter([*u],[yd-0.21,yd-0.21],color = col_ilr, marker = '|')
        
        
axdens[0].set(ylabel = '1',yticks = [])
axdens[1].set(xlabel = '$\pi(x)$',ylabel = '0',yticks = [],xlim  = (0, 1))

densfig.tight_layout()

rocfig,axroc = plt.subplots(1,1)

xl = []
xu = []
yl = []
yu = []
for i,j in zip(fpr_i,s_i):
    
    if not isinstance(i,pba.Interval):
        i = pba.I(i)
    if not isinstance(j,pba.Interval):
        j = pba.I(j)
      
    xl.append(i.left  )
    xu.append(i.right  )
    
    yl.append(j.left)
    yu.append(j.right)
    
axroc.plot(xl,yu, col_ilr,label = '$\mathcal{ILR}(F)$')
axroc.plot(xu,yl, col_ilr )
axroc.plot([0,1],[0,1],linestyle = ':',color=col_points)

axroc.set(xlabel = '$fpr$',ylabel='$s$')
axroc.plot(fpr,s,'k',label = '$\mathcal{LR}(D)$')
axroc.plot(nuq_fpr,nuq_s,color=col_mid,linestyle='--',label='$\mathcal{LR}(F_\\times)$')
axroc.plot(sslr_fpr,sslr_s,color='g',linestyle='--',label='$\mathcal{LR}(F_\\times)$')
axroc.plot(fpr_t,s_t,col_ilr2,label='$\mathcal{ILR}(F)$ (Predictive)')
axroc.legend()
# rocfig.savefig('figs/redwine_ROC.png',dpi = 600)
# rocfig.savefig('../LR-paper/figs/redwine_ROC.png',dpi = 600)
# densfig.savefig('figs/redwine_dens.png',dpi =600)
# densfig.savefig('../LR-paper/figs/redwine_dens.png',dpi =600)

tikzplotlib.save('figs/redwine_ROC.tikz',figure = rocfig,externalize_tables = True, override_externals = True,tex_relative_path_to_data = 'dat/redwine/')

tikzplotlib.save('figs/redwine_dens.tikz',figure = densfig,externalize_tables = False, override_externals = True,tex_relative_path_to_data = 'dat/redwine/')
print(*dat1,sep='\n',file = open('figs/dat/redwine/redwine_dens-000.dat','w'))
print(*dat2,sep='\n',file = open('figs/dat/redwine/redwine_dens-001.dat','w'))
print(*dat3,sep='\n',file = open('figs/dat/redwine/redwine_dens-002.dat','w'))
print(*dat4,sep='\n',file = open('figs/dat/redwine/redwine_dens-003.dat','w'))

#%%
with open('runinfo/redwine_auc.out','w') as f:
    print('NO UNCERTAINTY: %.4f' %auc(s,fpr), file = f)
    print('MIDPOINTS: %.4F' %auc(nuq_s,nuq_fpr),file = f)
    print('SSLR: %.4F' %auc(sslr_s,sslr_fpr),file = f)
    print('THROW: %.4f' %auc(s_t,fpr_t), file = f)
    print('ILR: [%.3f,%.3f]'  %(auc(yl,xu),auc(yu,xl)), file = f)
    
    # print('INTERVALS: [%.3f,%.3f]' %(auc_int_min,auc_int_max), file = f)
    
#%%
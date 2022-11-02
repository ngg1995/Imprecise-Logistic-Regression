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

from other_methods import DSLR, BDLR

# colors
col_precise = 'black'
col_points = '#A69888'
col_ilr = '#4169E1'
col_ilr2 = '#5d2e46'
col_ilr3 = '#FF8C00'
col_ilr4 = '#008000'
col_mid = '#DC143C'

def intervalise(val,eps,method,b=0.5,bounds = None):
    
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

    results = pd.Series(index = data.index, dtype = 'bool')
    
    for i in data.index:

        results.loc[i] = data.loc[i,0] >= 5+2*np.random.randn()    
        
    return results

def get_sample(data,r = None,ends = False, seed = 0):
    
    rng1 = np.random.default_rng(seed)
    n_data = data.copy()
    end = rng1.random(len(data)) > rng1.random(len(data))
    for c in data.columns:
        
        for i,e in zip(data.index,end):
            
            if data.loc[i,c].__class__.__name__ == 'Interval':
                
                if r is not None:
                    n_data.loc[i,c] = data.loc[i,c].left + r*data.loc[i,c].width()
                elif ends:
                    if e: 
                        n_data.loc[i,c] = data.loc[i,c].left
                    else:
                        n_data.loc[i,c] = data.loc[i,c].right
                else:
                    n_data.loc[i,c] = data.loc[i,c].left + np.random.random()*data.loc[i,c].width()
            
    return n_data

# %%
### Load Data
from dataset import train_data, train_results, test_data, test_results
#%%
### Fit logistic regression model
base = LogisticRegression()
base.fit(train_data.to_numpy(),train_results.to_numpy())

# Intervalise data
# set seed for reproducability
np.random.seed(10)
eps = 0.375

UQdata = pd.DataFrame({
    0:[intervalise(train_data.iloc[i,0],eps,'t',2/3,bounds = (0,10)) for i in train_data.index]
    }, dtype = 'O')

#%%
### Fit logistic regression model on full dataset
base = LogisticRegression(max_iter=1000)
base.fit(train_data.to_numpy(),train_results.to_numpy())

#%%
### Fit models with midpoint data
mid_data = midpoints(UQdata)
mid = LogisticRegression(max_iter=1000)
mid.fit(mid_data.to_numpy(),train_results.to_numpy())

#%%
### Fit de Souza model
ds = DSLR(max_iter = 1000)
ds.fit(UQdata,train_results)

#%%
### Fit Billard--Diday model
bd = BDLR(max_iter = 1000)
bd.fit(UQdata,train_results,N = 10000)
    
#%%
### Fit UQ models
ilr = ImpLogReg(uncertain_data=True, max_iter = 1000)
ilr.fit(UQdata,train_results,fast=False)
fast = ImpLogReg(uncertain_data=True, max_iter = 1000)
fast.fit(UQdata,train_results,fast=True,n_p_vals=1000)

# %% [markdown]
### Plots

def make_dat_file(fname,x,y):
    with open(f'figs/dat/features/{fname}.dat','w') as f:
        for i,j in zip(x,y):
            print(f"{i} {j}",file = f)

steps = 300
lX = np.linspace(0,10,steps)

lY = base.predict_proba(lX.reshape(-1, 1))[:,1]
make_dat_file('base',lX,lY)

lYm = mid.predict_proba(lX.reshape(-1, 1))[:,1]
make_dat_file('mid',lX,lYm)

lYd = ds.predict_proba(lX.reshape(-1,1))[:,1]
make_dat_file('deSouza',lX,lYd)

lYb = bd.predict_proba(lX.reshape(-1,1))[:,1]
make_dat_file('Billard-Diday',lX,lYb)

lYu = ilr.predict_proba(lX.reshape(-1,1))[:,1]
make_dat_file('minmaxcoed_right',lX,[i.right for i in lYu])
make_dat_file('minmaxcoed_left',lX,[i.left for i in lYu])

mc_min = np.full(len(lX),np.inf)
mc_max = np.full(len(lX),-np.inf)
for lr in bd:
    lY = lr.predict_proba(lX.reshape(-1, 1))[:,1]
    mc_min = np.minimum(mc_min.ravel(),lY.ravel())
    mc_max = np.maximum(mc_max.ravel(),lY.ravel())
make_dat_file('MC-right',lX,mc_min)
make_dat_file('MC-left',lX,mc_max)

lYu = fast.predict_proba(lX.reshape(-1,1))[:,1]
make_dat_file('fast_right',lX,[i.right for i in lYu])
make_dat_file('fast_left',lX,[i.left for i in lYu])

# %% [markdown]
### Get confusion matrix
# Classify test data
base_predict = base.predict(test_data)

# CLASSIFY NO_UQ MODEL DATA 
mid_predict = mid.predict(test_data)

# CLASSIFY UQ MODEL 
ilr_predict = ilr.predict(test_data)

with open('runinfo/features_cm.out','w') as f:
    print('TRUE MODEL',file = f)
    a,b,c,d = generate_confusion_matrix(test_results,base_predict)
    print('TP=%i\tFP=%i\nFN=%i\tTN=%i' %(a,b,c,d),file = f)

    # Calculate sensitivity and specificity
    print('Sensitivity = %.3f' %(a/(a+c)),file = f)
    print('Specificity = %.3f' %(d/(b+d)),file = f)

    print('DISCARDED DATA MODEL',file = f)
    aa,bb,cc,dd = generate_confusion_matrix(test_results,mid_predict)
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
    
    print('UQ MODEL',file = f)
    
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
    print('Sensitivity = [%.3f,%.3f]\nSpecificity = [%.3f,%.3f]' %(*sssi,*ttti),file = f)

    
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
mid_s,mid_fpr,mid_probabilities = ROC(model = mid, data = test_data, results = test_results)
s_t, fpr_t, Sigma, Tau = incert_ROC(ilr, test_data, test_results)

s_i, fpr_i,ilr_probabilities = ROC(ilr, test_data, test_results)

densfig,axdens = plt.subplots(nrows = 2, sharex= True)

dat1 = ['x y']
dat2 = ['x y']
dat3 = ['x y']
dat4 = ['x y']

for i,(p,u,midp,r) in enumerate(zip(probabilities,ilr_probabilities,mid_probabilities,test_results.to_list())):
    yd = np.random.uniform(-0.1,0.1)
    if r:
        dat1 += [f"{p} {yd}"]
        dat2 += [f"{midp} {0.21+yd}"]
        # axdens[0].scatter(p,yd,color = 'k',marker = 'o',alpha = 0.5)
        # axdens[0].scatter(midp,0.21+yd,color = col_mid,marker = 'o',alpha = 0.5)
        axdens[0].plot([*u],[yd-0.21,yd-0.21],color = col_ilr, alpha = 0.3)
        axdens[0].scatter([*u],[yd-0.21,yd-0.21],color = col_ilr, marker = '|') 
    else:
        dat3 += [f"{p} {yd}"]
        dat4 += [f"{midp} {0.21+yd}"]
        # axdens[1].scatter(p,yd,color = 'k',marker = 'o',alpha = 0.5)
        # axdens[1].scatter(midp,0.21+yd,color = col_mid,marker = 'o',alpha = 0.5)
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
    
axroc.plot(xl,yu, col_ilr,label = '$\mathcal{ILR}(E)$')
axroc.plot(xu,yl, col_ilr )
axroc.plot([0,1],[0,1],linestyle = ':',color=col_points)

axroc.set(xlabel = '$fpr$',ylabel='$s$')
axroc.plot(fpr,s,'k',label = '$\mathcal{LR}(D)$')
axroc.plot(mid_fpr,mid_s,color=col_mid,linestyle='--',label='$\mathcal{LR}(F_\\times)$')
axroc.plot(fpr_t,s_t,col_ilr2,label='$\mathcal{ILR}(E)$ (Predictive)')
axroc.legend()
# rocfig.savefig('figs/features_ROC.png',dpi = 600)
# rocfig.savefig('../LR-paper/figs/features_ROC.png',dpi = 600)
# densfig.savefig('figs/features_dens.png',dpi =600)
# densfig.savefig('../LR-paper/figs/features_dens.png',dpi =600)

tikzplotlib.save('figs/features_ROC.tikz',figure = rocfig,externalize_tables = True, override_externals = True,tex_relative_path_to_data = 'dat/features/')

tikzplotlib.save('figs/features_dens.tikz',figure = densfig,externalize_tables = False, override_externals = True,tex_relative_path_to_data = 'dat/features/')
print(*dat1,sep='\n',file = open('figs/dat/features/features_dens-000.dat','w'))
print(*dat2,sep='\n',file = open('figs/dat/features/features_dens-001.dat','w'))
print(*dat3,sep='\n',file = open('figs/dat/features/features_dens-002.dat','w'))
print(*dat4,sep='\n',file = open('figs/dat/features/features_dens-003.dat','w'))


with open('runinfo/features_auc.out','w') as f:
    print('NO UNCERTAINTY: %.4f' %auc(s,fpr), file = f)
    print('MIDPOINTS: %.4F' %auc(mid_s,mid_fpr),file = f)
    print('THROW: %.4f' %auc(s_t,fpr_t), file = f)
    print('ILR: [%.3f,%.3f]'  %(auc(yl,xu),auc(yu,xl)), file = f)
    
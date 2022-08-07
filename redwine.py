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

from LRF import *
from ImpLogReg import *
from other_methods import DSLR, BDLR


def midpoints(data):
    n_data = data.copy()
    for c in data.columns:
        for i in data.index:
            if data.loc[i,c].__class__.__name__ == 'Interval':

                n_data.loc[i,c] = data.loc[i,c].midpoint()

            
    return n_data

def intervalise(val,eps,method='u',b=0,bounds = None):
    
    if method == 'u':
        m = np.random.uniform(val-eps,val+eps)
    elif method == 't':
        m = np.random.triangular(val-eps,val-b*eps,val+eps)
    
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
### Import the data
redwine = pd.read_csv('redwine.csv',index_col = None)
X = redwine[[c for c in redwine.columns if c not in ['quality']]]
Y = redwine['quality'] >= 7

#%%
### Split into test and train samples
train_data, test_data, train_results, test_results = train_test_split(X, Y, test_size=0.9,random_state = 0,stratify=Y)
print(len(train_data),sum(train_results))
random.seed(1)
drop_index = random.sample([i for i in train_data.index if not train_results.loc[i]],k=2*sum(train_results))
train_data.drop(drop_index)
train_results.drop(drop_index)
UQdata = pd.DataFrame({
    "fixed acidity": [intervalise(i,0.1,'u',0.5,None) for i in train_data["fixed acidity"]],
    "volatile acidity": [intervalise(i,0.05,'u',0.75,None) for i in train_data["volatile acidity"]],
    "citric acid": [intervalise(i,0.01,'t',0,[0,1]) for i in train_data["citric acid"]],
    "residual sugar": [intervalise(i,0.1,'u',0,None) for i in train_data["residual sugar"]],
    "chlorides": [intervalise(i,0.001,'u',0,None) for i in train_data["chlorides"]],
    "free sulfur dioxide": [intervalise(i,1,'u',0,None) for i in train_data["free sulfur dioxide"]],
    "total sulfur dioxide": [intervalise(i,5,'u',0,None) for i in train_data["total sulfur dioxide"]],
    "density": [intervalise(i,0.002,'t',1,[0,1]) for i in train_data["density"]],
    "pH": [intervalise(i,0.05,'u',0,None) for i in train_data["pH"]],
    "sulphates": [intervalise(i,0.01,'u',0,None) for i in train_data["sulphates"]],
    "alcohol": [intervalise(i,0.2,"u",0,None) for i in train_data["alcohol"]]
    }, index = train_data.index, dtype = 'O'
)
#%%
### Fit logistic regression model on full dataset
base = LogisticRegression(max_iter=10000,solver='sag')
base.fit(train_data.to_numpy(),train_results.to_numpy())

#%%
### Fit models with midpoint data
mid_data = midpoints(UQdata)
mid = LogisticRegression(max_iter=10000,solver='sag')
mid.fit(mid_data.to_numpy(),train_results.to_numpy())

#%%
### Fit de Souza model
ds = DSLR(max_iter = 10000,solver='saga')
ds.fit(UQdata,train_results)

#%%
### Fit Billard--Diday model
bd = BDLR(max_iter = 10000,solver='saga')
bd.fit(UQdata,train_results)
    
#%%
### Fit UQ models
ilr = ImpLogReg(uncertain_data=True, max_iter = 10000,solver='saga')
ilr.fit(UQdata,train_results,fast = True)


# %% [markdown]
s,fpr,probabilities = ROC(model = base, data = test_data, results = test_results)
mid_s,mid_fpr,mid_probabilities = ROC(model = mid, data = test_data, results = test_results)
ds_s,ds_fpr,ds_probabilities = ROC(model = ds, data = test_data, results = test_results)
bd_s,bd_fpr,bd_probabilities = ROC(model = bd, data = test_data, results = test_results)
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
        axdens[0].plot([*u],[yd-0.21,yd-0.21],color = 'k', alpha = 0.3)
        axdens[0].scatter([*u],[yd-0.21,yd-0.21],color = 'k', marker = '|') 
    else:
        dat3 += [f"{p} {yd}"]
        dat4 += [f"{midp} {0.21+yd}"]
        # axdens[1].scatter(p,yd,color = 'k',marker = 'o',alpha = 0.5)
        # axdens[1].scatter(midp,0.21+yd,color = col_mid,marker = 'o',alpha = 0.5)
        axdens[1].plot([*u],[yd-0.21,yd-0.21],color = 'k', alpha = 0.3)
        axdens[1].scatter([*u],[yd-0.21,yd-0.21],color = 'k', marker = '|')
        
        
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
    
axroc.plot(xl,yu ,label = '$\mathcal{ILR}(E)$')
axroc.plot(xu,yl  )
axroc.plot([0,1],[0,1],linestyle = ':')

axroc.set(xlabel = '$fpr$',ylabel='$s$')
axroc.plot(fpr,s,'k',label = '$\mathcal{LR}(D)$')
axroc.plot(mid_fpr,mid_s,linestyle='--',label='$\mathcal{LR}(F_\\times)$')
axroc.plot(ds_fpr,ds_s,linestyle='--',label='ds')
axroc.plot(bd_fpr,bd_s,linestyle='--',label='bd')
axroc.plot(fpr_t,s_t,label='$\mathcal{ILR}(E)$ (Predictive)')
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


with open('runinfo/features_auc.out','w') as f:
    print('NO UNCERTAINTY: %.4f' %auc(s,fpr), file = f)
    print('MIDPOINTS: %.4F' %auc(mid_s,mid_fpr),file = f)
    print('THROW: %.4f' %auc(s_t,fpr_t), file = f)
    print('ILR: [%.3f,%.3f]'  %(auc(yl,xu),auc(yu,xl)), file = f)
#%%
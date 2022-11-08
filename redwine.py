#%%
import tikzplotlib
from ImpLogReg import ImpLogReg
import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
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
Y = redwine['quality'] >= 6
print(f"size = {len(X)}; good wine = {sum(Y)}")

#%%
### Split into test and train samples
train_data, test_data, train_results, test_results = train_test_split(X, Y, test_size=0.5,random_state = 0,stratify=Y)
print(f"train_sample = {len(train_data)} train_results = {sum(train_results)}")
# random.seed(1)
# drop_index = random.sample([i for i in train_data.index if not train_results.loc[i]],k=2*sum(train_results))
# train_data.drop(drop_index)
# train_results.drop(drop_index)
UQdata = pd.DataFrame({
    "fixed acidity": [intervalise(i,0.1,'u',0.5,None) for i in train_data["fixed acidity"]],
    "volatile acidity": [intervalise(i,0.01,'u',0.75,None) for i in train_data["volatile acidity"]],
    "citric acid": [intervalise(i,0.01,'t',0,[0,1]) for i in train_data["citric acid"]],
    "residual sugar": [intervalise(i,0.1,'u',0,None) for i in train_data["residual sugar"]],
    "chlorides": [intervalise(i,0.001,'u',0,None) for i in train_data["chlorides"]],
    "free sulfur dioxide": [intervalise(i,1,'u',0,None) for i in train_data["free sulfur dioxide"]],
    "total sulfur dioxide": [intervalise(i,1,'u',0,None) for i in train_data["total sulfur dioxide"]],
    "density": [intervalise(i,0.0001,'t',1,[0,1]) for i in train_data["density"]],
    "pH": [intervalise(i,0.01,'u',0,None) for i in train_data["pH"]],
    "sulphates": [intervalise(i,0.01,'u',0,None) for i in train_data["sulphates"]],
    "alcohol": [intervalise(i,0.1,"u",0,None) for i in train_data["alcohol"]]
    }, index = train_data.index, dtype = 'O'
)
#%%
### Fit logistic regression model on full dataset
base = LogisticRegression(max_iter=10000,solver='sag')
base.fit(train_data,train_results)

#%%
### Fit models with midpoint data
mid_data = midpoints(UQdata)
mid = LogisticRegression(max_iter=10000,solver='sag')
mid.fit(mid_data,train_results)

#%%
### Fit de Souza model
ds = DSLR(max_iter = 10000,solver='saga')
ds.fit(UQdata,train_results)

#%%
### Fit Billard--Diday model
bd = BDLR(max_iter = 10000,solver='saga')
bd.fit(UQdata,train_results,N = 10000)
    
#%%
### Fit UQ models
ilr = ImpLogReg(uncertain_data=True, max_iter = 10000,solver='saga')
ilr.fit(UQdata,train_results,fast = True,n_p_vals = 10)


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
        axdens[0].plot([*u],[yd-0.21,yd-0.21],color = 'k', marker = '|') 
    else:
        dat3 += [f"{p} {yd}"]
        dat4 += [f"{midp} {0.21+yd}"]
        # axdens[1].scatter(p,yd,color = 'k',marker = 'o',alpha = 0.5)
        # axdens[1].scatter(midp,0.21+yd,color = col_mid,marker = 'o',alpha = 0.5)
        axdens[1].plot([*u],[yd-0.21,yd-0.21],color = 'k', marker = '|')
        
        
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


with open('runinfo/redwine_auc.out','w') as f:
    print('NO UNCERTAINTY: %.4f' %auc(s,fpr), file = f)
    print('MIDPOINTS: %.4F' %auc(mid_s,mid_fpr),file = f)
    print('bd: %.4F' %auc(bd_s,bd_fpr),file = f)
    print('ds: %.4F' %auc(ds_s,ds_fpr),file = f)
    print('ILR: [%.3f,%.3f]'  %(auc(yl,xu),auc(yu,xl)), file = f)
    print('THROW: %.4f' %auc(s_t,fpr_t), file = f)
    
# %%
### Get confusion matrix
C = 0.55
# Classify test data
base_predict = probabilities>=C

# CLASSIFY NO_UQ MODEL DATA 
mid_predict = mid_probabilities>=C

# CLASSIFY NO_UQ MODEL DATA 
bd_predict = bd_probabilities>=C

# CLASSIFY NO_UQ MODEL DATA 
ds_predict = ds_probabilities>=C

# CLASSIFY UQ MODEL 
ilr_predict = [i>= C for i in ilr_probabilities]

with open('runinfo/redwine_cm.out','w') as f:
    print('TRUE MODEL',file = f)
    a,b,c,d = generate_confusion_matrix(test_results,base_predict)
    print('TP=%i\tFP=%i\nFN=%i\tTN=%i' %(a,b,c,d),file = f)

    # Calculate sensitivity and specificity
    print('Sensitivity = %.3f' %(a/(a+c)),file = f)
    print('Specificity = %.3f' %(d/(b+d)),file = f)

    print('\nmid MODEL',file = f)
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

    print('\nDS MODEL',file = f)
    aa,bb,cc,dd = generate_confusion_matrix(test_results,ds_predict)
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
    
    print('\nbd MODEL',file = f)
    aa,bb,cc,dd = generate_confusion_matrix(test_results,bd_predict)
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
   
# %%

from ImpLogReg import ImpLogReg
import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
import itertools as it
from tqdm import tqdm
import pba

from LRF import *
from ImpLogReg import *

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

### Import the data
wine_data = pd.read_csv('redwine.csv',index_col = None)

### Split the data into test/train factors and result and generate uncertain points
random.seed(2) # for reproducability

train_data_index = random.sample(list(wine_data.index), k = 200)
test_data_index = [i for i in wine_data.index if i not in train_data_index]


cols = [c for c in wine_data.columns if c in ["citric acid","total sulfur dioxide","sulphates","alcohol"]]

# test_data = wine_data.loc[test_data_index,cols]
# test_results = wine_data.loc[test_data_index,"quality"] >= 6

train_data = wine_data.loc[train_data_index,cols]
train_results = wine_data.loc[train_data_index,"quality"] >= 6

uq_results_index = random.sample([i for i in train_data_index if wine_data.loc[i,"quality"] == 6], k = 5)
uq_results = pd.Series([pba.I(0,1) if i in uq_results_index else train_results.loc[i] for i in train_data_index],index = train_data_index, dtype='O')

# intervalise(val,eps,method='u',b=0,bounds = None):

uq_data = pd.DataFrame({
    # "fixed acidity": [intervalise(i,0.1,'u',0.5,None) for i in train_data["fixed acidity"]],
    # "volatile acidity": [intervalise(i,0.05,'u',0.75,None) for i in train_data["volatile acidity"]],
    "citric acid": [intervalise(i,0.01,'t',0,[0,1]) for i in train_data["citric acid"]],
    # "residual sugar": [intervalise(i,0.1,'u',0,None) for i in train_data["residual sugar"]],
    # "chlorides": [intervalise(i,0.001,'u',0,None) for i in train_data["chlorides"]],
    # "free sulfur dioxide": [intervalise(i,1,'u',0,None) for i in train_data["free sulfur dioxide"]],
    "total sulfur dioxide": [intervalise(i,5,'u',0,None) for i in train_data["total sulfur dioxide"]],
    # "density": [intervalise(i,0.002,'t',1,[0,1]) for i in train_data["density"]],
    # "pH": [intervalise(i,0.05,'u',0,None) for i in train_data["pH"]],
    "sulphates": [intervalise(i,0.01,'u',0,None) for i in train_data["sulphates"]],
    "alcohol": [intervalise(i,0.2,'u',0,None) for i in train_data["alcohol"]]
    }, index = train_data_index, dtype = 'O'
)

nuq_data = deintervalise(uq_data.loc[[i for i in train_data_index if i not in uq_results_index]],[None])
nuq_results = pd.Series([train_results.loc[i] for i in train_data_index if i not in uq_results_index])

### Fit logistic regression model on full dataset
base = LogisticRegression(max_iter=1000)
base.fit(train_data.to_numpy(),train_results.to_numpy())

### Fit nuq model
nuq = LogisticRegression(max_iter=1000)
nuq.fit(nuq_data.to_numpy(),nuq_results.to_numpy())

### Fit uq model
ilr = ImpLogReg(max_iter = 1000, uncertain_class=True, uncertain_data=True)
ilr.fit(uq_data, uq_results)


### Get confusion matrix
# Classify test data
base_predict = base.predict(test_data)

# CLASSIFY NO_UQ MODEL DATA 
nuq_predict = nuq.predict(test_data)

# CLASSIFY UQ MODEL 
ilr_predict = ilr.predict(test_data)

with open('runinfo/redwine_cm.out','w') as f:
    print('TRUE MODEL',file = f)
    a,b,c,d = generate_confusion_matrix(test_results,base_predict)
    print('TP=%i\tFP=%i\nFN=%i\tTN=%i' %(a,b,c,d),file = f)

    # Calculate sensitivity and specificity
    print('Sensitivity = %.3f' %(a/(a+c)),file = f)
    print('Specificity = %.3f' %(d/(b+d)),file = f)

    print('DISCARDED DATA MODEL',file = f)
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
   

# ### Descriminatory Performance Plots
# s,fpr,probabilities = ROC(model = base, data = train_data, results = train_results)

# nuq_s,nuq_fpr,nuq_probabilities = ROC(model = nuq, data = train_data, results = train_results)
# s_t, fpr_t, Sigma, Tau = incert_ROC(ilr, train_data, train_results)

# s_i, fpr_i,ilr_probabilities = ROC(ilr, train_data, train_results)

# densfig,axdens = plt.subplots(nrows = 2, sharex= True)

# for i,(p,u,nuqp,r) in enumerate(zip(probabilities,ilr_probabilities,nuq_probabilities,train_results.to_list())):
#     yd = np.random.uniform(-0.1,0.1)
#     if r:
#         axdens[0].scatter(p,yd,color = 'k',marker = 'o',alpha = 0.5)
#         axdens[0].scatter(nuqp,0.21+yd,color = '#DC143C',marker = 'o',alpha = 0.5)
#         axdens[0].plot([*u],[yd-0.21,yd-0.21],color = '#4169E1',alpha = 0.3)
#         axdens[0].scatter([*u],[yd-0.21,yd-0.21],color = '#4169E1',marker = '|')
#     else:
#         axdens[1].scatter(p,yd,color = 'k',marker = 'o',alpha = 0.5)
#         axdens[1].scatter(nuqp,0.21+yd,color = '#DC143C',marker = 'o',alpha = 0.5)
#         axdens[1].plot([*u],[yd-0.21,yd-0.21],color = '#4169E1',alpha = 0.3)
#         axdens[1].scatter([*u],[yd-0.21,yd-0.21],color = '#4169E1',marker = '|')
        
        
# axdens[0].set(ylabel = 'Outcome = 1',yticks = [])
# axdens[1].set(xlabel = '$\pi(x)$',ylabel = 'Outcome = 0',yticks = [],xlim  = (0, 1))

# densfig.tight_layout()

# rocfig,axroc = plt.subplots(1,1)
# axroc.plot([0,1],[0,1],'k:',label = 'Random Classifier')
# axroc.set(xlabel = '$fpr$',ylabel='$s$')
# axroc.plot(fpr,s,'k',label = 'Base')
# axroc.plot(nuq_fpr,nuq_s,color='#DC143C',linestyle='--',label='Ignored Uncertainty')
# axroc.plot(fpr_t,s_t,'#4169E1',label='Imprecise Model')
# axroc.legend()
# rocfig.savefig('figs/redwine_ROC.png',dpi = 600)
# rocfig.savefig('../paper/figs/redwine_ROC.png',dpi = 600)
# densfig.savefig('figs/redwine_dens.png',dpi =600)
# densfig.savefig('../paper/figs/redwine_dens.png',dpi =600)


# with open('runinfo/redwine_auc.out','w') as f:
#     print('NO UNCERTAINTY: %.3f' %auc(s,fpr), file = f)
#     print('MIDPOINTS: %.4F' %auc(nuq_s,nuq_fpr),file = f)
#     print('THROW: %.3f' %auc(s_t,fpr_t), file = f)
#     # print('INTERVALS: [%.3f,%.3f]' %(auc_int_min,auc_int_max), file = f)
    


# fig = plt.figure()
# ax = plt.axes(projection='3d',elev = 45,azim = -45,proj_type = 'ortho')
# ax.set_xlabel('$fpr$')
# ax.set_ylabel('$s$')
# # ax.set_zlabel('$1-\sigma,1-\\tau$')
# ax.plot(fpr_t,s_t,'#4169E1',alpha = 0.5)
# ax.plot3D(fpr_t,s_t,Sigma,'#FF8C00',label = '$\\sigma$')
# ax.plot3D(fpr_t,s_t,Tau,'#008000',label = '$\\tau$')
# # ax.plot3D(fpr,s,Nu,'k',label = '$1-\\nu$')

# ax.legend()

# plt.savefig('figs/redwine_ROC3D.png',dpi = 600)
# plt.savefig('../paper/figs/redwine_ROC3D.png',dpi = 600)
# plt.clf()

# plt.xlabel('$fpr$/$s$')
# plt.ylabel('$\\sigma$/$\\tau$')
# plt.plot(s_t,Sigma,'#FF8C00',label = '$\\sigma$ v $s$')
# plt.plot(fpr_t,Tau,'#008000',label = '$\\tau$ v $fpr$')
# plt.legend()


# plt.savefig('figs/redwine_ST.png',dpi = 600)
# plt.savefig('../paper/figs/redwine_ST.png',dpi = 600)

# plt.clf()


# ### Hosmer-Lemeshow
# hl_b, pval_b = hosmer_lemeshow_test(base,train_data,train_results,g = 10)

# hl_nuq, pval_nuq = hosmer_lemeshow_test(nuq,train_data,train_results,g = 10)
# #
# hl_uq, pval_uq = UQ_hosmer_lemeshow_test(ilr,train_data,train_results,g = 10)

# with open('runinfo/redwine_HL.out','w') as f:
#     print('base\nhl = %.3f, p = %.3f' %(hl_b,pval_b),file = f)
#     print('no UQ\nhl = %.3f, p = %.3f' %(hl_nuq,pval_nuq),file = f) 

#     print('UQ\nhl = [%.3f,%.3f], p = [%.3f,%.3f]' %(*hl_uq,*pval_uq),file = f) 
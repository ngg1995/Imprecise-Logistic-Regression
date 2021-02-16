import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
import itertools as it
from tqdm import tqdm
import pba
import tikzplotlib
import random

from LRF import *

def intervalise(val,eps,method,b=0.5,bounds = None):
    
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
eps = 0.5

UQdata = pd.DataFrame({
    0:[intervalise(train_data.iloc[i,0],eps,'t',-.5,(0,10)) for i in train_data.index]
    }, dtype = 'O')

### Fit logistic regression model on full dataset
base = LogisticRegression(max_iter=1000)
base.fit(train_data.to_numpy(),train_results.to_numpy())


### Fit models with midpoint data
nuq_data = midpoints(UQdata)
nuq = LogisticRegression(max_iter=1000)
nuq.fit(nuq_data.to_numpy(),train_results.to_numpy())

### Fit UQ models
uq_models = int_logistic_regression(UQdata,train_results)


### Plot results
steps = 300
lX = np.linspace(0,10,steps)
lY = base.predict_proba(lX.reshape(-1, 1))[:,1]
lYn = nuq.predict_proba(lX.reshape(-1, 1))[:,1]

plt.xlabel('$x$')
plt.ylabel('$\pi_x$')
plt.plot(lX,lY,color='k',zorder=10,lw=2)
plt.plot(lX,lYn,color='m',zorder=10,lw=2)

# for u,m,r in zip(UQdata[0],train_data[0],train_results.to_list()):
#     plt.plot(m,r,marker = 'x')
#     plt.plot([u.Left,u.Right],[r,r], marker='|')

lYmin = np.ones(steps)
lYmax = np.zeros(steps)

for n, model in uq_models.items():
    lY = model.predict_proba(np.linspace(0,10,steps).reshape(-1, 1))[:,1]
    lYmin = [min(i,j) for i,j in zip(lY,lYmin)]
    lYmax = [max(i,j) for i,j in zip(lY,lYmax)]
    plt.plot(lX,lY,color = 'grey',alpha = 0.2,lw = 0.5)

plt.plot(lX,lYmax,color='red',lw=2)
plt.plot(lX,lYmin,color='red',lw=2)

plt.savefig('../paper/figs/ex1_int.png',dpi = 600)
plt.savefig('figs/ex1_int.png',dpi = 600)

plt.clf()


### Get confusion matrix
# Classify test data
base_predict = base.predict(test_data)

# CLASSIFY NO_UQ MODEL DATA 
nuq_predict = nuq.predict(test_data)

# CLASSIFY UQ MODEL 
test_predict = pd.DataFrame(columns = uq_models.keys())

for key, model in uq_models.items():
    test_predict[key] = model.predict(test_data)
    
predictions = []
for i in test_predict.index:
    predictions.append([min(test_predict.loc[i]),max(test_predict.loc[i])])

with open('runinfo/ex1_int_cm.out','w') as f:
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
    
    aaai,bbbi,ccci,dddi = generate_confusion_matrix(test_results,predictions,throw = False)
    try:
        sssi = 1/(1+ccci/aaai)
    except:
        sssi = None
    try:    
        ttti = 1/(1+bbbi/dddi)
    except:
        ttti = None
        
    print('TP=%s\tFP=%s\nFN=%s\tTN=%s' %(aaai,bbbi,ccci,dddi),file = f)

    # Calculate sensitivity and specificity
    print('Sensitivity = %s' %(sssi),file = f)
    print('Specificity = %s' %(ttti),file = f)
    
    aaa,bbb,ccc,ddd,eee,fff = generate_confusion_matrix(test_results,predictions,throw = True)
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


# ### ROC CURVE
s,fpr = ROC(model = base, data = test_data, results = test_results)
nuq_s,nuq_fpr = ROC(model = nuq, data = test_data, results = test_results)
s_t, fpr_t, Sigma, Tau, Nu = UQ_ROC_alt(uq_models, test_data, test_results)

plt.xlabel('$1-t$')
plt.ylabel('$s$')
plt.step(fpr,s,'k', label = 'Base')
plt.step(nuq_fpr,nuq_s,'m', label = 'Discarded')
plt.step(fpr_t,s_t,'r', label = 'Not Predicting')
plt.legend()

plt.savefig('figs/ex1_int_ROC.png',dpi = 600)
plt.savefig('../paper/figs/ex1_int_ROC.png',dpi = 600)
plt.clf()

with open('runinfo/ex1_int_auc.out','w') as f:
    print('NO UNCERTAINTY: %.4f' %auc(s,fpr), file = f)
    # print('NO UNCERTAINTY: %.4f' %roc_auc_score(base.predict_proba(test_data)[:,1],test_results), file = f)
    # print('LOWER BOUND: %.4f' %auc(Ymin,Xmin), file = f)
    # print('UPPER BOUND: %.4f' %auc(Ymax,Xmax), file = f)
    print('THROW: %.4f' %auc(s_t,fpr_t), file = f)

fig = plt.figure()

ax = plt.axes(projection='3d',elev = 45,azim = -45,proj_type = 'ortho')
ax.set_xlabel('$1-t$')
ax.set_ylabel('$s$')
ax.set_zlabel('$\\sigma,\\tau$')
ax.plot(fpr_t,s_t,'m',alpha = 0.5)
ax.plot3D(fpr,s,Sigma,'b',label = '$\\sigma$')
ax.plot3D(fpr,s,Tau,'r',label = '$\\tau$')
# ax.plot3D(fpr,s,Nu,'k',label = '$1-\\nu$')

ax.legend()

plt.savefig('figs/ex1_int_ROC3D.png',dpi = 600)
plt.savefig('../paper/figs/ex1_int_ROC3D.png',dpi = 600)

### Hosmer-Lemeshow
hl_b, pval_b = hosmer_lemeshow_test(base,train_data,train_results,Q = 20)

hl_nuq, pval_nuq = hosmer_lemeshow_test(nuq,train_data,train_results,Q = 20)

hl_min = np.inf
hl_max = 0
pval_max = 0
pval_min = 1

for k,m in uq_models.items():
    hl_,pval_ = hosmer_lemeshow_test(m,train_data,train_results,Q = 10)
    hl_min = min(hl_min,hl_)
    pval_min = min(pval_min,pval_)
    
    hl_max = max(hl_max,hl_)
    pval_max = max(pval_max,pval_)
    
    
with open('runinfo/ex1_int_HL.out','w') as f:
    print('base\nhl = %.3f, p = %.5f' %(hl_b,pval_b),file = f)
    print('no UQ\nhl = %.3f, p = %.5f' %(hl_nuq,pval_nuq),file = f) 

    print('UQ\nhl = [%.3f,%.3f], p = [%.5f,%.5f]' %(hl_min,hl_max,pval_min,pval_max),file = f) 
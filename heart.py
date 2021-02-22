import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import itertools as it
from tqdm import tqdm
import pba
import random

from LRF import *

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

heart_data = pd.read_csv('SAheart.csv',index_col = 'patient')


# Split the data into test/train factors and result
random.seed(1111) # for reproducability
np.random.seed(1)

train_data_index = random.sample([i for i in heart_data.index ], k = 150)
test_data_index = [i for i in heart_data.index if i not in train_data_index]

test_data = heart_data.loc[test_data_index,[c for c in heart_data.columns if c != 'chd']]
train_data = heart_data.loc[train_data_index,[c for c in heart_data.columns if c != 'chd']]

test_results = heart_data.loc[test_data_index,'chd']
train_results = heart_data.loc[train_data_index,'chd']

# Intervalise data
eps = {'sbp':(1,'u'),
       'tobacco':(0.1,'t',-1),
       'ldl': (0.01,'u'),
       'adiposity': (0.01,'u'),
    #    'famhist',
       'typea':(1,'u'),
       'obesity':(0.01,'u'),
       'alcohol':(0.1,'t',-1),
       'age': (-1,'b',0)
       }

np.random.seed(0)
UQdata = pd.DataFrame({
    **{k:[intervalise(train_data.loc[i,k],*eps[k]) for i in train_data.index] for k, e in eps.items()},
    **{c:train_data[c] for c in train_data.columns if c not in eps.keys()}
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

with open('runinfo/heart_cm.out','w') as f:
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
    print('Sensitivity = [%.3f,%.3f]\nSpecificity = [%.3f,%.3f]' %(*sssi,*ttti),file = f)

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


### ROC CURVE
s,fpr = ROC(model = base, data = test_data, results = test_results)
nuq_s,nuq_fpr = ROC(model = nuq, data = test_data, results = test_results)
s_t, fpr_t, Sigma, Tau, Nu = UQ_ROC_alt(uq_models, test_data, test_results)

s_i, fpr_i = UQ_ROC(uq_models, test_data, test_results)

steps = 1000
X = np.linspace(0,1,steps)
Ymin = steps*[2]
Ymax = steps*[-1]

for i, x in tqdm(enumerate(X)):
    for k,j in zip(s_i,fpr_i):

        if j.straddles(x,endpoints = True):
            Ymin[i] = min((Ymin[i],k.Left))
            Ymax[i] = max((Ymax[i],k.Right))

Xmax = [0]+[x for i,x in enumerate(X) if Ymax[i] != -1]+[1]
Xmin = [0]+[x for i,x in enumerate(X) if Ymin[i] != 2]+[1]
Ymax = [0]+[y for i,y in enumerate(Ymax) if Ymax[i] != -1]+[1]
Ymin = [0]+[y for i,y in enumerate(Ymin) if Ymin[i] != 2]+[1]

auc_int_min = sum([(Xmin[i]-Xmin[i-1])*Ymin[i] for i in range(1,len(Xmin))])
auc_int_max = sum([(Xmax[i]-Xmax[i-1])*Ymax[i] for i in range(1,len(Xmin))])
   
plt.xlabel('$1-t$')
plt.ylabel('$s$')
plt.ylabel('$\\sigma,\\tau$')

plt.step(fpr,s,'k', label = 'Base')
plt.step(nuq_fpr,nuq_s,'m--', label = 'Discarded')
plt.step(fpr_t,s_t,'y', label = 'Not Predicting')
plt.plot(Xmax,Ymax,'r',label = 'Interval Bounds')
plt.plot(Xmin,Ymin,'r')

plt.legend()

plt.savefig('figs/heart_ROC.png',dpi = 600)
plt.savefig('../paper/figs/heart_ROC.png',dpi = 600)
# plt.clf()

# for i,j in zip(fpr_i,s_i):
#     plt.plot([i.Left,i.Right],[j.Left,j.Right])
# plt.show()
with open('runinfo/heart_auc.out','w') as f:
    print('NO UNCERTAINTY: %.4f' %auc(s,fpr), file = f)
    print('DISCARDED: %.4F' %auc(nuq_s,nuq_fpr),file = f)
    print('THROW: %.4f' %auc(s_t,fpr_t), file = f)
    print('INTERVALS: [%.4f,%.4f]' %(auc_int_min,auc_int_max), file = f)
    

fig = plt.figure()

ax = plt.axes(projection='3d',elev = 45,azim = -45,proj_type = 'ortho')
ax.set_xlabel('$1-t$')
ax.set_ylabel('$s$')
# ax.set_zlabel('$1-\sigma,1-\\tau$')
ax.plot(fpr_t,s_t,'m',alpha = 0.5)
ax.plot3D(fpr,s,Sigma,'b',label = '$\\sigma$')
ax.plot3D(fpr,s,Tau,'r',label = '$\\tau$')
# ax.plot3D(fpr,s,Nu,'k',label = '$1-\\nu$')

ax.legend()

plt.savefig('figs/heart_ROC3D.png',dpi = 600)
plt.savefig('../paper/figs/heart_ROC3D.png',dpi = 600)
plt.clf()

plt.xlabel('$(1-t)$/$s$')
plt.ylabel('$\\sigma$/$\\tau$')
plt.plot(s,Sigma,'g',label = '$\\sigma$ v $s$')
plt.plot(fpr,Tau,'r',label = '$\\tau$ v $t$')
plt.legend()

plt.savefig('figs/heart_ST.png',dpi = 600)
plt.savefig('../paper/figs/heart_ST.png',dpi = 600)

plt.clf()

### Hosmer-Lemeshow
hl_b, pval_b = hosmer_lemeshow_test(base,train_data,train_results,g = 10)

hl_nuq, pval_nuq = hosmer_lemeshow_test(nuq,train_data,train_results,g = 10)

hl_uq, pval_uq = UQ_hosmer_lemeshow_test(uq_models,train_data,train_results,g = 10)

    
    
with open('runinfo/heart_HL.out','w') as f:
    print('base\nhl = %.3f, p = %.5f' %(hl_b,pval_b),file = f)
    print('no UQ\nhl = %.3f, p = %.5f' %(hl_nuq,pval_nuq),file = f) 
    print('UQ\nhl = [%.3f,%.3f], p = [%.3f,%.3f]' %(*hl_uq,*pval_uq),file = f) 
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.linear_model import LogisticRegression
import itertools as it
from tqdm import tqdm
import pba
import random

from LRF import *

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

### Fit logistic regression model on full dataset
base = LogisticRegression()
base.fit(train_data.to_numpy(),train_results.to_numpy())

### Remove uncertain datapoints from the training data
few = 5 #uncertain points
random.seed(5) # for reproducability
uq_data_index = random.sample([i for i in train_data.index if abs(train_data.loc[i,0]-5) <= 1.5], k = few) # clustered around center

uq_data = train_data.loc[uq_data_index]
nuq_data = train_data.loc[[i for i in train_data.index if i not in uq_data_index]]
nuq_results = train_results.loc[[i for i in train_data.index if i not in uq_data_index]]

### Fit UQ models
uq_models = uc_logistic_regression(train_data,train_results,uq_data)

### Fit models with missing data
nuq = LogisticRegression()
nuq.fit(nuq_data.to_numpy(),nuq_results.to_numpy())

### Plot results
steps = 300
lX = np.linspace(0,10,steps)
lY = base.predict_proba(lX.reshape(-1, 1))[:,1]
lYn = nuq.predict_proba(lX.reshape(-1, 1))[:,1]

plt.xlabel('$x$')
plt.ylabel('$\pi_x$')
plt.scatter(nuq_data,nuq_results,color='blue',zorder=10)
plt.plot(lX,lY,color='k',zorder=10,lw=2,label = 'Truth')
plt.plot(lX,lYn,color='m',zorder=10,lw=2,label = 'No UQ')

for x in uq_data[0]:

    plt.plot([x,x],[0,1],color='blue')

lYmin = np.ones(steps)
lYmax = np.zeros(steps)

for n, model in uq_models.items():
    lY = model.predict_proba(np.linspace(0,10,steps).reshape(-1, 1))[:,1]
    lYmin = [min(i,j) for i,j in zip(lY,lYmin)]
    lYmax = [max(i,j) for i,j in zip(lY,lYmax)]
    plt.plot(lX,lY,color = 'grey',alpha = 0.2,lw = 0.5)

plt.plot(lX,lYmax,color='red',lw=2)
plt.plot(lX,lYmin,color='red',lw=2,label = 'Uncertainty Bounds')

plt.savefig('../paper/figs/ex1_UC.png',dpi = 600)
plt.savefig('figs/ex1_UC.png',dpi = 600)

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

with open('runinfo/ex1_UC_cm.out','w') as f:
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
        
    print('TP=[%i,%i]\tFP=[%i,%i]\nFN=[%i,%i]\tTN=[%i,%i]' %(*aaai,*bbbi,*ccci,*dddi),file = f)

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
    print('sigma = %.3f' %(eee/(aaa+ccc+eee)),file = f)
    print('tau = %.3f' %(fff/(bbb+ddd+fff)),file = f)
    

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
plt.step(fpr,s,'k', label = 'Base')
plt.step(nuq_fpr,nuq_s,'m--', label = 'Discarded')
plt.step(fpr_t,s_t,'y', label = 'Not Predicting')
plt.plot(Xmax,Ymax,'r',label = 'Interval Bounds')
plt.plot(Xmin,Ymin,'r')

plt.legend()

plt.savefig('figs/ex1_UC_ROC.png',dpi = 600)
plt.savefig('../paper/figs/ex1_UC_ROC.png',dpi = 600)
plt.clf()

with open('runinfo/ex1_UC_auc.out','w') as f:
    print('NO UNCERTAINTY: %.3f' %auc(s,fpr), file = f)
    print('DISCARDED: %.4F' %auc(nuq_s,nuq_fpr),file = f)
    print('THROW: %.3f' %auc(s_t,fpr_t), file = f)
    print('INTERVALS: [%.3f,%.3f]' %(auc_int_min,auc_int_max), file = f)
    

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

plt.savefig('figs/ex1_UC_ROC3D.png',dpi = 600)
plt.savefig('../paper/figs/ex1_UC_ROC3D.png',dpi = 600)
plt.clf()

plt.xlabel('$(1-t)$/$s$')
plt.ylabel('$\\sigma$/$\\tau$')
plt.plot(s,Sigma,'g',label = '$\\sigma$ v $s$')
plt.plot(fpr,Tau,'r',label = '$\\tau$ v $t$')
plt.legend()

plt.savefig('figs/ex1_UC_ST.png',dpi = 600)
plt.savefig('../paper/figs/ex1_UC_ST.png',dpi = 600)

plt.clf()

### Hosmer-Lemeshow
hl_b, pval_b = hosmer_lemeshow_test(base,train_data,train_results,g = 10)

hl_nuq, pval_nuq = hosmer_lemeshow_test(nuq,train_data,train_results,g = 10)
   
hl_uq, pval_uq = UQ_hosmer_lemeshow_test(uq_models,train_data,train_results,g = 10)

with open('runinfo/ex1_UC_HL.out','w') as f:
    print('base\nhl = %.3f, p = %.3f' %(hl_b,pval_b),file = f)
    print('no UQ\nhl = %.3f, p = %.3f' %(hl_nuq,pval_nuq),file = f) 

    print('UQ\nhl = [%.3f,%.3f], p = [%.3f,%.3f]' %(*hl_uq,*pval_uq),file = f) 
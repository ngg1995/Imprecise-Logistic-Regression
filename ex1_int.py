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

def generate_results(data):

    results = pd.Series(index = data.index, dtype = 'bool')
    for row in data.index:

        results[row] = sum(data.loc[row]) >= len(data.columns)*(15+5*np.random.randn())
    
    return results

def intervalise(val,eps):

    m = np.random.uniform(val-eps,val+eps)
    
    return pba.I(m-eps,m+eps)

def midpoints(data):
    n_data = data.copy()
    for c in data.columns:
        for i in data.index:
            if data.loc[i,c].__class__.__name__ == 'Interval':
                n_data.loc[i,c] = data.loc[i,c].midpoint()-5
            
    return n_data

# set seed to ensure same data
np.random.seed(10)

# Params
many = 50
dim = 1
some = 10000
eps = 2

# Generate data
data = pd.DataFrame(40*np.random.rand(many,dim))
results = generate_results(data)

# Intervalise data
UQdata = pd.DataFrame({
    0:[intervalise(data.iloc[i,0],eps) for i in data.index]
    }, dtype = 'O')

# Generate test data 
np.random.seed(111)
test_data = pd.DataFrame(40*np.random.rand(some,dim))
test_results = generate_results(test_data)

# Fit true model
truth = LogisticRegression(max_iter = 1000)
truth.fit(data.to_numpy(),results.to_numpy())

# Fit base model
# Base model is midpoints
MDdata = midpoints(UQdata)
base = LogisticRegression(max_iter = 1000)
base.fit(MDdata.to_numpy(),results.to_numpy())

# Classify test data
truth_predict = truth.predict(test_data)
base_predict = base.predict(test_data)

## fit interval model
uq_models = int_logistic_regression(UQdata,results)

## Test estimated vs Monte Carlo
ir, oor = check_int_MC(uq_models,UQdata,results,1000,test_data)
with open('runinfo/ex1_int_MCtest.out','w') as f:
    print('in bounds %i,%.3f\nout %i,%.3f'%(ir,(ir/(ir+oor)),oor,(oor/(ir+oor))),file = f)

# Classify test data
test_predict = pd.DataFrame(columns = uq_models.keys())

for key, model in uq_models.items():
    test_predict[key] = model.predict(test_data)
    
predictions = []
for i in test_predict.index:
    predictions.append([min(test_predict.loc[i]),max(test_predict.loc[i])])


# # Plot results
steps = 1000
lX = np.linspace(data.min(),data.max(),steps)
lYb = base.predict_proba(lX.reshape(-1, 1))[:,1]
lYt = truth.predict_proba(lX.reshape(-1, 1))[:,1]


plt.xlabel('X')
plt.ylabel('$\Pr(Y=1|X)$')

for u,m,r in zip(UQdata[0],data[0],results.to_list()):
    plt.plot(m,r,marker = 'x')
    plt.plot([u.Left,u.Right],[r,r], marker='|')
plt.plot(lX,lYb,color='c',zorder=10,lw=2,label = 'Midpoint')
plt.plot(lX,lYt,color='k',zorder=10,lw=2,label = 'Truth')

lYmin = np.ones(steps)
lYmax = np.zeros(steps)

for n, model in uq_models.items():

    lY = model.predict_proba(np.linspace(data.min(),data.max(),steps).reshape(-1, 1))[:,1]
    lYmin = [min(i,j) for i,j in zip(lY,lYmin)]
    lYmax = [max(i,j) for i,j in zip(lY,lYmax)]


    # plt.plot(lX,lY,color = 'grey')


plt.plot(lX,lYmax,color='red',lw=2)
plt.plot(lX,lYmin,color='red',lw=2)

plt.savefig('../paper/figs/ex1_int.png',dpi = 600)
plt.savefig('figs/ex1_int.png',dpi = 600)

plt.clf()

## Get confusion matrix
with open('runinfo/ex1_int_cm.out','w') as f:
    
    print('~~~~TRUE MODEL~~~~', file = f)
    a,b,c,d = generate_confusion_matrix(test_results,truth_predict)
    print('TP=%i\tFP=%i\nFN=%i\tTN=%i' %(a,b,c,d),file = f)
    # Calculate sensitivity and specificity
    print('Sensitivity = %.3f' %(a/(a+c)),file = f)
    print('Specificity = %.3f' %(d/(b+d)),file = f)
    
    print('~~~~BASE MODEL~~~~', file = f)
    a,b,c,d = generate_confusion_matrix(test_results,base_predict)
    print('TP=%i\tFP=%i\nFN=%i\tTN=%i' %(a,b,c,d),file = f)
    # Calculate sensitivity and specificity
    print('Sensitivity = %.3f' %(a/(a+c)),file = f)
    print('Specificity = %.3f' %(d/(b+d)),file = f)

    print('~~~~UQ MODEL~~~~', file = f)
    aaa,bbb,ccc,ddd,eee,fff = generate_confusion_matrix(test_results,predictions,throw = True)
    print('TP=%i\tFP=%i\nFN=%i\tTN=%i\nNP(+)=%i\tNP(-)=%i' %(aaa,bbb,ccc,ddd,eee,fff),file = f)
    try:
        sss = 1/(1+ccc/aaa)
        print('Sensitivity = %.3f' %(sss),file = f) 
    except:
        pass
    try:    
        ttt = 1/(1+bbb/ddd)
        print('Specificity = %.3f' %(ttt),file = f)
    except:
        pass

    print('sigma = %3f' %(eee/(aaa+ccc+eee)),file = f)
    print('tau = %3f' %(fff/(bbb+ddd+fff)),file = f)

### ROC CURVE
s_b,fpr_b = ROC(model = base, data = test_data, results = test_results)
s,fpr = ROC(model = truth, data = test_data, results = test_results)
s_t, fpr_t, Sigma, Tau, Nu = UQ_ROC_alt(uq_models, test_data, test_results)
with open('roc.txt','w') as f:
    for i,j in zip(zip(s_b,fpr_b),zip(s,fpr)):
        print(i,j,file =f)
plt.plot([0,1],[0,1],'k:',label = 'Random Classifer')
plt.xlabel('$1-t$')
plt.ylabel('$s$')
plt.plot(fpr,s,'k', marker = 'x',label = 'Truth')
plt.plot(fpr_b,s_b,'c',marker = 'o', label = 'Midpoint')

steps = 1001
X = np.linspace(0,1,steps)
Ymin = steps*[2]
Ymax = steps*[-1]

plt.plot(fpr_t,s_t,'m', label = 'IP model')
plt.legend()

plt.savefig('figs/ex1_int_ROC.png',dpi = 600)
plt.savefig('../paper/figs/ex1_int_ROC.png',dpi = 600)

plt.clf()

with open('runinfo/ex1_int_auc.out','w') as f:
    print('Truth: %.4f' %auc(s,fpr), file = f)
    print('Midpoints: %.4f' %auc(s_b,fpr_b), file = f)
    print('IP: %.4f' %auc(s_t,fpr_t), file = f)

######
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

plt.savefig('figs/ex1_int_ROC3D.png',dpi = 600)
plt.savefig('../paper/figs/ex1_int_ROC3D.png',dpi = 600)
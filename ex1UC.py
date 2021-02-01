import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics as skm
import itertools as it
from tqdm import tqdm
import pba
import tikzplotlib
from LRF import *

def generate_results(data):

    results = pd.Series(index = data.index, dtype = 'bool')
    for row in data.index:

        results[row] = sum(data.loc[row]) >= len(data.columns)*(15+5*np.random.randn())
    
    return results


# set seed to ensure same data
np.random.seed(10)

# Params
many = 25
dim = 1
few = 5
some = 100

# Generate data
data = pd.DataFrame(40*np.random.rand(many,dim))
results = generate_results(data)

# Generate uncertain points
uncertain = 15+pd.DataFrame(5*np.random.rand(few,dim))

# Generate test data 
np.random.seed(111)
test_data = pd.DataFrame(40*np.random.rand(some,dim))
test_results = generate_results(test_data)

# Fit base model
base = LogisticRegression()
base.fit(data.to_numpy(),results.to_numpy())

# Classify test data
base_predict = base.predict(test_data)

# Fit model
uq_models = uc_logistic_regression(data,results,uncertain)

# Classify test data
test_predict = pd.DataFrame(columns = uq_models.keys())

for key, model in uq_models.items():
    test_predict[key] = model.predict(test_data)
    
predictions = []
for i in test_predict.index:
    predictions.append([min(test_predict.loc[i]),max(test_predict.loc[i])])

# Fit base model
base = LogisticRegression()
base.fit(data.to_numpy(),results.to_numpy())

# Classify test data
base_predict = base.predict(test_data)
print(base.coef_)

# # Plot results
lX = np.linspace(data.min(),data.max(),100)
lY = base.predict_proba(lX.reshape(-1, 1))[:,1]

plt.xlabel('X')
plt.ylabel('$\Pr(Y=1|X)$')
plt.scatter(data,results,color='blue',zorder=10)
plt.plot(lX,lY,color='k',zorder=10,lw=2)
plt.savefig('../paper/figs/ex1.png')
plt.savefig('figs/ex1.png')


for x in uncertain[0]:

    plt.plot([x,x],[0,1],color='blue')

lYmin = np.ones(300)
lYmax = np.zeros(300)

for n, model in uq_models.items():
    lY = model.predict_proba(np.linspace(data.min(),data.max(),100).reshape(-1, 1))[:,1]
    lYmin = [min(i,j) for i,j in zip(lY,lYmin)]
    lYmax = [max(i,j) for i,j in zip(lY,lYmax)]
    plt.plot(lX,lY,color = 'grey')


plt.plot(lX,lYmax,color='red',lw=2)
plt.plot(lX,lYmin,color='red',lw=2)

plt.savefig('../paper/figs/ex1_UC.png')
plt.savefig('figs/ex1_UC.png')


plt.clf()

## Get confusion matrix
with open('ex1-cm.out','w') as f:
    a,b,c,d = generate_confusion_matrix(test_results,base_predict)
    print('TP=%i\tFP=%i\nFN=%i\tTN=%i' %(a,b,c,d),file = f)

    # Calculate sensitivity and specificity
    print('Sensitivity = %.3f' %(a/(a+c)),file = f)
    print('Specificity = %.3f' %(d/(b+d)),file = f)


    aa,bb,cc,dd = generate_confusion_matrix(test_results,predictions)
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
    print('Sensitivity = %s' %(ss),file = f)
    print('Specificity = %s' %(tt),file = f)

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
s_i, fpr_i, s_t, fpr_t = UQ_ROC(models = uq_models, data = test_data, results = test_results)

plt.plot([0,1],[0,1],'k:',label = '$s = 1-t$')
plt.plot([0,0,1],[0,1,1],'r:',label = '$s = 1-t$')
plt.xlabel('$1-t$')
plt.ylabel('$s$')
plt.step(fpr,s,'k', label = 'Base')
plt.savefig('figs/ex1_ROC.png')
plt.savefig('../paper/figs/ex1_ROC.png')


steps = 1001
X = np.linspace(0,1,steps)
Ymin = steps*[2]
Ymax = steps*[-1]

for i, x in enumerate(X):
    for k,j in zip(s_i,fpr_i):
        plt.plot([j.Left,j.Left,j.Right,j.Right,j.Left],[k.Left,k.Right,k.Right,k.Left,k.Left],c= 'grey')
        if j.straddles(x,endpoints = True):
            Ymin[i] = min((Ymin[i],k.Left))
            Ymax[i] = max((Ymax[i],k.Right))
            
plt.step([x for i,x in enumerate(X) if Ymax[i] != -1],[y for i,y in enumerate(Ymax) if Ymax[i] != -1],'r',label = 'Upper Bound')
plt.step([x for i,x in enumerate(X) if Ymin[i] != 2],[y for i,y in enumerate(Ymin) if Ymin[i] != 2],'b',label = 'Lower Bound')

plt.step(fpr_t,s_t,'m', label = 'Dropped Values')
plt.legend()

# tikzplotlib.save('../paper/figs/ex1_UQ_ROC.png')
plt.savefig('figs/ex1_UQ_ROC.png')
plt.savefig('../paper/figs/ex1_UQ_ROC.png')

# plt.clf()

with open('ex1-auc.out','w') as f:
    print('NO UNCERTAINTY: %.4f' %auc(s,fpr), file = f)
    print('LOWER BOUND: %.4f' %auc([x for i,x in enumerate(X) if Ymin[i] != 2],[y for i,y in enumerate(Ymin) if Ymin[i] != 2]), file = f)
    print('UPPER BOUND: %.4f' %auc([x for i,x in enumerate(X) if Ymax[i] != 2],[y for i,y in enumerate(Ymax) if Ymax[i] != 2]), file = f)
    print('THROW: %.4f' %auc(s_t,fpr_t), file = f)

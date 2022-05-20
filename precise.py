import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
import itertools as it
from tqdm import tqdm
import pba
import random
from LRF import *

import matplotlib
font = {'size'   : 14,'family' : 'Times New Roman'}
matplotlib.rc('font', **font)

col_precise = 'black'
col_points = 'grey'
col_ilr = '#4169E1'
col_mid = '#DC143C'

def generate_results(data):
    # set seed for reproducability
    np.random.seed(10)
    results = pd.Series(index = data.index, dtype = 'bool')
    
    for i in data.index:

        results.loc[i] = data.loc[i,0] >= 5+2*np.random.randn()    
        
    return results


### Generate Data
# set seed for reproducability
s = 1234
np.random.seed(s)
random.seed(s)

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

### Plot results
steps = 300
lX = np.linspace(0,10,steps)
lY = base.predict_proba(lX.reshape(-1, 1))[:,1]

plt.xlabel('$x$')
plt.ylabel('$\pi(x)$')
plt.scatter(train_data,train_results,color='grey',zorder=10)
# plt.scatter(test_data,test_results,color='green',zorder=10)
plt.plot(lX,lY,color='k',zorder=10,lw=2)
plt.savefig('figs/precise.png',dpi = 600)
plt.savefig('../LR-paper/figs/precise.png',dpi = 600)
plt.clf()

### Get confusion matrix 
base_predict = base.predict(test_data)
print(len(test_results))
with open('runinfo/precise_cm.out','w') as f:
    a,b,c,d = generate_confusion_matrix(test_results,base_predict)
    print('TP=%i\tFP=%i\nFN=%i\tTN=%i' %(a,b,c,d),file = f)

    # Calculate sensitivity and specificity
    print('Sensitivity = %.3f' %(a/(a+c)),file = f)
    print('Specificity = %.3f' %(d/(b+d)),file = f)


### ROC/AUC
s,fpr, predictions = ROC(model = base, data = test_data, results = test_results)

rocfig,axroc = plt.subplots(1)
densfig,axdens = plt.subplots(1)
axdens.scatter(predictions,test_results+np.random.uniform(-0.05,0.05,len(predictions)),marker = 'o',color='k',alpha = 0.5)
axdens.set(xlabel = '$\pi(x)$',ylabel = 'Outcome',yticks = [0,1],xlim  = (0, 1))


axroc.plot([0,1],[0,1],'k:',label = 'Random Classifier')
axroc.set(xlabel = '$fpr$',ylabel='$s$')
axroc.plot(fpr,s,'k')

rocfig.savefig('figs/precise_ROC.png',dpi = 600)
rocfig.savefig('../LR-paper/figs/precise_ROC.png',dpi = 600)
densfig.savefig('figs/precise_dens.png',dpi = 600)
densfig.savefig('../LR-paper/figs/precise_dens.png',dpi = 600)

with open('runinfo/precise_auc.out','w') as f:
    print('AUC: %.4f' %auc(s,fpr), file = f)


### Hosmer-Lemeshow
hl, pval = hosmer_lemeshow_test(base,train_data,train_results,g = 10)
with open('runinfo/precise_HL.out','w') as f:
    print('hl = %.3f, p = %.5f' %(hl,pval),file = f) 

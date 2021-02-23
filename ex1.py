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

### Plot results
steps = 300
lX = np.linspace(0,10,steps)
lY = base.predict_proba(lX.reshape(-1, 1))[:,1]

plt.xlabel('$x$')
plt.ylabel('$\pi_x$')
plt.scatter(train_data,train_results,color='blue',zorder=10)
# plt.scatter(test_data,test_results,color='green',zorder=10)
plt.plot(lX,lY,color='k',zorder=10,lw=2)
plt.savefig('figs/ex1.png',dpi = 600)
plt.savefig('../paper/figs/ex1.png',dpi = 600)
plt.clf()

### Get confusion matrix 
base_predict = base.predict(test_data)
print(len(test_results))
with open('runinfo/ex1_cm.out','w') as f:
    a,b,c,d = generate_confusion_matrix(test_results,base_predict)
    print('TP=%i\tFP=%i\nFN=%i\tTN=%i' %(a,b,c,d),file = f)

    # Calculate sensitivity and specificity
    print('Sensitivity = %.3f' %(a/(a+c)),file = f)
    print('Specificity = %.3f' %(d/(b+d)),file = f)


### ROC/AUC
s,fpr = ROC(model = base, data = test_data, results = test_results)
plt.plot([0,1],[0,1],'k:',label = 'Random Classifier')
plt.plot([0],[1],'ro',label = 'Perfect Classifier')
plt.xlabel('$fpr$')
plt.ylabel('$s$')
plt.plot(fpr,s,'k', label = 'Model')
plt.legend()

plt.savefig('figs/ex1_ROC.png',dpi = 600)
plt.savefig('../paper/figs/ex1_ROC.png',dpi = 600)

with open('runinfo/ex1_auc.out','w') as f:
    print('AUC: %.4f' %auc(s,fpr), file = f)


### Hosmer-Lemeshow
hl, pval = hosmer_lemeshow_test(base,train_data,train_results,g = 10)
with open('runinfo/ex1_HL.out','w') as f:
    print('hl = %.3f, p = %.5f' %(hl,pval),file = f) 

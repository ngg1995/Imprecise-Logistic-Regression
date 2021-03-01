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
s,fpr, predictions = ROC(model = base, data = test_data, results = test_results)

rocfig,ax = plt.subplots(2,2)

ax[0,0].scatter(predictions,test_results+np.random.uniform(-0.05,0.05,len(predictions)),marker = '.',color='k')
ax[0,0].set(xlabel = '$\pi$',ylabel = 'Outcome')


ax[1,0].plot([0,1],[0,1],'k:',label = 'Random Classifier')
ax[1,0].set(xlabel = '$fpr$',ylabel='$s$')
ax[1,0].plot(fpr,s,'k')

ax[0,1].hist([p for p,r in zip(predictions,test_results) if r],bins = 20,color='k')
ax[0,1].set(title = 'Outcome = 1',xlabel = '$\pi$',ylabel = 'Density')
ax[0,1].yaxis.set_label_position("right")
ax[0,1].yaxis.tick_right()

ax[1,1].hist([p for p,r in zip(predictions,test_results) if not r],bins = 20,color='k')
ax[1,1].set(title = 'Outcome = 0',xlabel = '$\pi$',ylabel = 'Density')
ax[1,1].yaxis.set_label_position("right")
ax[1,1].yaxis.tick_right()

rocfig.tight_layout()
rocfig.savefig('figs/ex1_ROC.png',dpi = 600)
rocfig.savefig('../paper/figs/ex1_ROC.png',dpi = 600)

with open('runinfo/ex1_auc.out','w') as f:
    print('AUC: %.4f' %auc(s,fpr), file = f)


### Hosmer-Lemeshow
hl, pval = hosmer_lemeshow_test(base,train_data,train_results,g = 10)
with open('runinfo/ex1_HL.out','w') as f:
    print('hl = %.3f, p = %.5f' %(hl,pval),file = f) 

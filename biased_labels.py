import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
import pba
import random

from ImpLogReg import *
from LRF import *


def generate_results(data):
    # set seed for reproducability
    np.random.seed(10)
    results = pd.Series(index = data.index, dtype = 'bool')
    
    for i in data.index:

        results.loc[i] = 0.5*np.cos(np.pi/6 * data.loc[i,0]) + 0.5 >= np.random.rand()
        
    return results


### Generate Data
# set seed for reproducability
np.random.seed(258)

# Params
some = 50 # training datapoints

data = pd.DataFrame(10*np.random.rand(some,1))
results = generate_results(data)

### Fit logistic regression model on full dataset
base = LogisticRegression()
base.fit(data.to_numpy(),results.to_numpy())

### Remove uncertain datapoints from the training data
few = 5 #uncertain points
random.seed(1) # for reproducability
uq_data_index = random.sample([i for i in data.index if data.loc[i,0] >= 7], k = few) # clustered around end

uq_data = data.loc[uq_data_index]
uq_results = pd.Series([int(results.loc[i]) if i not in uq_data_index else pba.I(0,1) for i in results.index], index = results.index, dtype='O')
nuq_data = data.loc[[i for i in data.index if i not in uq_data_index]]
nuq_results = results.loc[[i for i in data.index if i not in uq_data_index]]

### Fit UQ models
ilr = ImpLogReg(uncertain_class=True, max_iter = 1000)
ilr.fit(data,uq_results)


### Fit models with missing data
nuq = LogisticRegression()
nuq.fit(nuq_data.to_numpy(),nuq_results.to_numpy())

### Plot results
steps = 300
lX = np.linspace(0,10,steps)
lY = base.predict_proba(lX.reshape(-1, 1))[:,1]
lYn = nuq.predict_proba(lX.reshape(-1, 1))[:,1]
lYu = ilr.predict_proba(lX.reshape(-1,1))[:,1]

plt.xlabel('$x$')
plt.ylabel('$\pi(x)$')
plt.scatter(nuq_data,nuq_results,color='grey',zorder=10)
# plt.plot(lX,lY,color='k',zorder=10,lw=2,label = 'Truth')
plt.plot(lX,lYn,color='#DC143C',zorder=10,lw=2,label = 'No UQ')

for i in uq_data_index:

    plt.plot([uq_data.loc[i],uq_data.loc[i]],[0,1],color='grey')
    plt.scatter(uq_data.loc[i],results.loc[i],marker = 'd',color = 'black',zorder = 14)
    
plt.plot(lX,[i.left for i in lYu],color='#4169E1',lw=2)
plt.plot(lX,[i.right for i in lYu],color='#4169E1',lw=2,label = 'Uncertainty Bounds')

plt.savefig('../LR-paper/figs/biased_labels.png',dpi = 600)
plt.savefig('figs/biased_labels.png',dpi = 600)
# plt.show()
# plt.clf()




### Hosmer-Lemeshow
hl_b, pval_b = hosmer_lemeshow_test(base,data,results,g = 10)

hl_nuq, pval_nuq = hosmer_lemeshow_test(nuq,data,results,g = 10)
#
hl_uq, pval_uq = UQ_hosmer_lemeshow_test(ilr,data,results,g = 10)

with open('runinfo/biased_UC_HL.out','w') as f:
    print('base\nhl = %.3f, p = %.3f' %(hl_b,pval_b),file = f)
    print('no UQ\nhl = %.3f, p = %.3f' %(hl_nuq,pval_nuq),file = f) 

    print('UQ\nhl = [%.3f,%.3f], p = [%.3f,%.3f]' %(*hl_uq,*pval_uq),file = f) 
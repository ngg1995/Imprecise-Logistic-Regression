import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import SelfTrainingClassifier
import itertools as it
from tqdm import tqdm
import pba
import random

import matplotlib
font = {'size'   : 14}
matplotlib.rc('font', **font)


def histogram(probs, dif = 0.05, uq = False, bars = 10):
    x = np.arange(bars)/bars
    if uq: 
        low_height = bars*[0]
        hi_height = bars*[0]
    else:
        height = bars*[0]
    for p in probs:
        for i,j in reversed(list(enumerate(x))):
            if uq:
                if i + 1 == bars:
                    if p[0] > j:
                        low_height[i] += 1/len(probs)
                        hi_height[i] += 1/len(probs)
                        break
                    if p[1] > j:
                        hi_height[i] += 1/len(probs)
                else:
                    if p[0] > j and p[1] < x[i+1]:
                        low_height[i] += 1/len(probs)
                        hi_height[i] += 1/len(probs)
                        break
                    if p[1] > j:
                        hi_height[i] += 1/len(probs)
                    if p[0] > j:
                        hi_height[i] += 1/len(probs)
                        break
            else:
                if p > j:
                    height[i] += 1/len(probs)
                    break
            
    if dif != 0:
        x = [i+dif for i in x]    
    
    if uq:
        return x,low_height, hi_height
    return x, height

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

# uq_data = train_data.loc[uq_data_index]
uq_results = pd.Series([int(train_results.loc[i]) if i not in uq_data_index else -1 for i in train_results.index], index = train_results.index)
nuq_data = train_data.loc[[i for i in train_data.index if i not in uq_data_index]]
nuq_results = train_results.loc[[i for i in train_data.index if i not in uq_data_index]]

### Fit UQ models
uq_models = SelfTrainingClassifier(LogisticRegression())
uq_models.fit(train_data,uq_results)

### Fit models with missing data
nuq = LogisticRegression()
nuq.fit(nuq_data.to_numpy(),nuq_results.to_numpy())

### Plot results
steps = 300
lX = np.linspace(0,10,steps)
lY = base.predict_proba(lX.reshape(-1, 1))[:,1]
lYn = nuq.predict_proba(lX.reshape(-1, 1))[:,1]
lYu = uq_models.predict_proba(lX.reshape(-1, 1))[:,1]

plt.xlabel('$x$')
plt.ylabel('$\pi(x)$')
plt.scatter(nuq_data,nuq_results,color='grey',zorder=10)
plt.plot(lX,lY,color='k',zorder=10,lw=2,label = 'Truth')
plt.plot(lX,lYn,color='#DC143C',zorder=10,lw=2,label = 'No UQ')
plt.plot(lX,lYu,color='#ABCDEF',zorder=10,lw=2,label = 'SSL')

plt.show()
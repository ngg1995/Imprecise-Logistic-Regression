import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
import itertools as it
from tqdm import tqdm
import pba
import tikzplotlib

def generate_results(data):

    results = pd.Series(index = data.index, dtype = 'bool')
    for row in data.index:

        results[row] = sum(data.loc[row]) >= len(data.columns)*(15+3*np.random.randn())
    
    return results

def generate_confusion_matrix(results,predictions,throw = False):

    a = 0
    b = 0
    c = 0
    d = 0
    
    for result, prediction in zip(results,predictions):
        if prediction.__class__.__name__ != 'list':
            prediction = [prediction,prediction]

        if prediction[0] == prediction[1]:
            if result:
                if prediction[0]:
                    a += 1
                else:
                    c += 1
            else:
                if prediction[0]:
                    b += 1
                else:
                    d += 1
        elif not throw:
            if result:
                a += pba.I(0,1)
                c += pba.I(0,1)
            else:
                b += pba.I(0,1)
                d += pba.I(0,1)
                    
    return a,b,c,d

def find_threshold(model):
    X = np.linspace(0,30,1000000)
    for i,j in zip(X,model.predict(X.reshape(-1,1))):
        if j:
            return i

def get_bounds(UQdata,results):
    bounds = {
        'minimum':[],    
        'maximum': []
    }

    for i in UQdata:
        bounds['minimum'] += [i.Left]
        # maximum
        bounds['maximum'] += [i.Right]


    models = {}
    # predict from bounds
    for n, b in bounds.items():
        d1 = np.array(b).reshape(-1,1)

        model = LogisticRegression()
        models[n] = model.fit(d1,results.to_numpy())

    minThreshold = find_threshold(models['minimum'])
    maxThreshold = find_threshold(models['maximum'])

    bounds2 = {        
        'minTs': [],
        'maxTs': [],
        'minTm': [],
        'maxTm': []
        }
    for i in UQdata:
        if abs(minThreshold - i.Left) > abs(minThreshold - i.Right):
            bounds2['minTs'] += [i.Left]
        else:
            bounds2['minTs'] += [i.Right]

        if abs(maxThreshold - i.Left) > abs(maxThreshold - i.Right):
            bounds2['maxTs'] += [i.Left]
        else:
            bounds2['maxTs'] += [i.Right]

        if i.straddles(maxThreshold):
            bounds2['maxTm'] += [maxThreshold]
        elif abs(maxThreshold - i.Left) < abs(maxThreshold - i.Right):
            bounds2['maxTm'] += [i.Left]
        else:
            bounds2['maxTm'] += [i.Right]

        if i.straddles(minThreshold):
            bounds2['minTm'] += [minThreshold]
        elif abs(minThreshold - i.Left) > abs(minThreshold - i.Right):
            bounds2['minTm'] += [i.Left]
        else:
            bounds2['minTm'] += [i.Right]

    # predict from bounds
    for n, b in bounds2.items():
        d1 = np.array(b).reshape(-1,1)
        model = LogisticRegression()
        models[n] = model.fit(d1,results.to_numpy())

    return models


# set seed to ensure same data
np.random.seed(10)


# Params
many = 25
dim = 1
steps = 100
some = 100

# Generate data
data = pd.DataFrame(30*np.random.rand(many,dim))
results = generate_results(data)
eps = 2

UQdata = [pba.I(data.iloc[i,0]-eps*np.random.rand(),data.iloc[i,0]+eps*np.random.rand()) for i in data.index]

# Generate test data 
test_data = pd.DataFrame(30*np.random.rand(some,dim))
test_results = generate_results(test_data)


for u,r in zip(UQdata,results.to_list()):
    plt.plot([u.Left,u.Right],[r,r],marker='|')

# Fit base model
base = LogisticRegression()
base.fit(data.to_numpy(),results.to_numpy())
# Classify test data
base_predict = base.predict(test_data)

# fit int model
models = get_bounds(UQdata,results)

lX = np.linspace(0,30,steps).reshape(-1, 1)
lY = base.predict_proba(lX)[:,1]
plt.plot(lX,lY,color='k',lw=2)

lYmin = np.ones(steps)
lYmax = np.zeros(steps)

for n, model in models.items():
    lY = model.predict_proba(lX)[:,1]
    lYmin = [min(i,j) for i,j in zip(lY,lYmin)]
    lYmax = [max(i,j) for i,j in zip(lY,lYmax)]

    # plt.plot(lX,lY,label=n)

plt.plot(lX,lYmax,color='red',lw=2)
plt.plot(lX,lYmin,color='red',lw=2)

plt.show()
tikzplotlib.save('paper/figs/ex1intLReps.tikz')


# Get confusion matrix
a,b,c,d = generate_confusion_matrix(test_results,base_predict)
try:
    s = 1/(1+c/a)
except:
    s = None
try:    
    t = 1/(1+b/d)
except:
    t = None
print(a,b,c,d,s,t)
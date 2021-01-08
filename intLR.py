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

        results[row] = sum(data.loc[row]) >= len(data.columns)*(15+2*np.random.rand())
    
    return results

def find_threshold(model):
    X = np.linspace(0,30,1000000)
    for i,j in zip(X,model.predict(X.reshape(-1,1))):
        if j:
            return i

def get_bounds(UQdata,results):
    bounds = {
        'minimum':[],    
        'maximum': [],
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
np.random.seed(11)

# Params
many = 25
dim = 1
steps = 1000
eps = 0.1

# Generate data
data = pd.DataFrame(30*np.random.rand(many,dim))
results = generate_results(data)

UQdata = [pba.I(data.iloc[i,0]-eps,data.iloc[i,0]+eps) for i in data.index]

# Fit base model
base = LogisticRegression()
base.fit(data.to_numpy(),results.to_numpy())

models = get_bounds(UQdata,results)

lYmin = np.ones(steps)
lYmax = np.zeros(steps)
lX = np.linspace(0,30,steps)
for n, model in models.items():
    lY = model.predict_proba(np.linspace(data.min(),data.max(),steps).reshape(-1, 1))[:,1]
    lYmin = [min(i,j) for i,j in zip(lY,lYmin)]
    lYmax = [max(i,j) for i,j in zip(lY,lYmax)]
    plt.plot(lX,lY,label=n)
plt.legend()

# plt.plot(lX,lYmax,color='red',lw=2)
# plt.plot(lX,lYmin,color='red',lw=2)

# models = {}
# weird_data = {}
# N = 0
# M = 0
# for i,r in tqdm(enumerate(it.product([-eps,0,eps],repeat=many)),total=3**len(data)):
#     new_data = data[0]+r
#     # print(type(new_data))
#     model = LogisticRegression()
#     model.fit(new_data.to_numpy().reshape(-1, 1),results.to_numpy())
#     lY = model.predict_proba(np.linspace(data.min(),data.max(),steps).reshape(-1, 1))[:,1]
#     for ii,jj,kk,ll in zip(lY,lYmax,lYmin,np.linspace(data.min(),data.max(),steps)):
#         M+=1
#         if ii > jj or ii < kk:
#             # print(ii,jj)
#             # print(ii,kk)
#             # print(ll)
#             N+=1
#             weird_data[r] = new_data 
#             # plt.plot(lX,lY,color = 'grey')

# print(N)
# print(M)
# # pd.DataFrame(weird_data).to_csv('weird_data.csv')
plt.show() 



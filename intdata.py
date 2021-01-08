import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
import itertools as it
from tqdm import tqdm
import pba

def generate_results(data):

    results = pd.Series(index = data.index, dtype = 'bool')
    for row in data.index:

        results[row] = sum(data.loc[row]) >= len(data.columns)*(15+2*np.random.rand())
    
    return results

def int_logistic_regression(data,result,eps):

    models = {}

    for i,r in tqdm(enumerate(it.product([-eps,eps],repeat=many)),total=2**len(data)):
        new_data = data[0]+r
        # print(type(new_data))
        model = LogisticRegression()       
        models[str(i)] = model.fit(new_data.to_numpy().reshape(-1, 1),results.to_numpy())
        
    return models

def intm_logistic_regression(data,result,eps):

    models = {}

    for i,r in tqdm(enumerate(it.product([-eps,0,eps],repeat=many)),total=3**len(data)):
        new_data = data[0]+r
        # print(type(new_data))
        model = LogisticRegression()       
        models[str(i)] = model.fit(new_data.to_numpy().reshape(-1, 1),results.to_numpy())
        
    return models

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

# set seed to ensure same data
np.random.seed(11)

# Params
many = 8

dim = 1
some = 10
eps = 0.1

# Generate data
data = pd.DataFrame(30*np.random.rand(many,dim))
results = generate_results(data)

# # Generate test data 
# test_data = pd.DataFrame(3*np.random.rand(some,dim))
# test_results = generate_results(test_data)

# Fit model
# models = int_logistic_regression(data,results,eps)
# modelsM = intm_logistic_regression(data,results,eps)


# Classify test data
# test_predict = pd.DataFrame(columns = models.keys())

# for key, model in models.items():
#     test_predict[key] = model.predict(test_data)
    
# predictions = []
# for i in test_predict.index:
#     predictions.append([min(test_predict.loc[i]),max(test_predict.loc[i])])

# Fit base model
# base = LogisticRegression()
# base.fit(data.to_numpy(),results.to_numpy())

# # Classify test data
# base_predict = base.predict(test_data)

for d,r in zip(data.to_numpy(),results.to_numpy()):
    plt.plot([d+eps,d-eps],[r,r],marker='|')

# X = np.linspace(0,3,100)
# lYmin = np.ones(300)
# lYmax = np.zeros(300)
# for n, model in models.items():
#     lY = model.predict_proba(np.linspace(data.min(),data.max(),100).reshape(-1, 1))[:,1]
#     lYmin = [min(i,j) for i,j in zip(lY,lYmin)]
#     lYmax = [max(i,j) for i,j in zip(lY,lYmax)]
#     # plt.plot(X,lY,color = 'grey')


# plt.plot(X,lYmax,color='red',lw=2)
# plt.plot(X,lYmin,color='red',lw=2)

# lYmin = np.ones(300)
# lYmax = np.zeros(300)
# for n, model in modelsM.items():
#     lY = model.predict_proba(np.linspace(data.min(),data.max(),100).reshape(-1, 1))[:,1]
#     lYmin = [min(i,j) for i,j in zip(lY,lYmin)]
#     lYmax = [max(i,j) for i,j in zip(lY,lYmax)]
#     # plt.plot(X,lY,color = 'grey')


# plt.plot(X,lYmax,color='green',lw=2)
# plt.plot(X,lYmin,color='green',lw=2)
plt.show()
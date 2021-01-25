import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import itertools as it
from tqdm import tqdm
import pba
import random

def generate_confusion_matrix(results,predictions,throw = False):

    a = 0
    b = 0
    c = 0
    d = 0
    e = 0
    f = 0
    
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
        else:
            if result:
                e += 1
            else:
                f += 1
    
    if throw:
        return a,b,c,d,e,f
    else:
        return a,b,c,d

def uc_logistic_regression(data,result,uncertain):

    models = {}

    for N,i in tqdm(enumerate(it.product([0,1],repeat=len(uncertain))),total=2**len(uncertain)):

        new_data = pd.concat((data,uncertain), ignore_index = True)
        new_result = pd.concat((result, pd.Series(i)), ignore_index = True)

        model = LogisticRegression(max_iter=1000)       
        models[str(i)] = model.fit(new_data.to_numpy(),new_result.to_numpy())
        
    return models

def ROC(model = None, predictions = None, data = None, results = None):
    
    s = []
    fpr = []
    
    if predictions is None:
        predictions = model.predict_proba(data)[:,1]
    
    for p in np.linspace(0,1,101):
        a = 0
        b = 0
        c = 0
        d = 0

        for prediction, result in zip(predictions,results):

            if prediction >= p:
                if result:
                    # true positive
                    a += 1
                else:
                    # false positive
                    b+= 1
            else: 
                if result:
                    # false negative
                    c += 1
                else:
                    # true negative
                    d += 1
                    
        
        s.append(a/(a+c))
        fpr.append(b/(b+d))
    return s, fpr
   
def UQ_ROC(models, data, results):
    
    s = []
    fpr = []
    
    predictions_lb = []
    predictions_ub = []
    
    for d in tqdm(data.index):
        l = [m.predict_proba(data.loc[d].to_numpy().reshape(1, -1))[:,1] for k,m in models.items()]
        predictions_lb += [min(l)]
        predictions_ub += [max(l)]
        
    s_lb,fpr_lb = ROC(predictions = predictions_lb, data = data, results = results)
    s_ub,fpr_ub = ROC(predictions = predictions_ub, data = data, results = results)
        
    return s_lb, fpr_lb, s_ub, fpr_ub

def split_data(features, results, test_frac = 0.5, uq_frac = 0.05, seed=random.random()):
    
    i = list(features.index)
    n = len(i)
   
    # get data indexes
    test_data_index = random.sample(i, k = int(n*test_frac))
    train_data_index = random.sample([f for f in i if f not in test_data_index], k = int((1-uq_frac) * (n-len(test_data_index))))
    uq_data_index = [f for f in i if f not in test_data_index and f not in train_data_index]

    test_data = features.loc[test_data_index]
    train_data = features.loc[train_data_index]
    uq_data = features.loc[uq_data_index]
    print('%i test data\n%i training data\n%i uncertain data' %(len(test_data_index),len(train_data_index),len(uq_data_index)))
    test_results = results.loc[test_data_index]
    train_results = results.loc[train_data_index]
  
    
    return test_data, test_results, train_data, train_results, uq_data

__all__ = [
    'generate_confusion_matrix',
    'uc_logistic_regression',
    'ROC',
    'UQ_ROC',
    'split_data'
]
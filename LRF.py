import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import itertools as it
from tqdm import tqdm
import pba
import random

def midpoints(data):
    n_data = data.copy()
    for c in data.columns:
        for i in data.index:
            if data.loc[i,c].__class__.__name__ == 'Interval':
                n_data.loc[i,c] = data.loc[i,c].midpoint()
            
    return n_data


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

    for N,i in tqdm(enumerate(it.product([0,1],repeat=len(uncertain))),total=2**len(uncertain),desc='UC Logistic Regression'):

        new_data = pd.concat((data,uncertain), ignore_index = True)
        new_result = pd.concat((result, pd.Series(i)), ignore_index = True)

        model = LogisticRegression(max_iter=1000)       
        models[str(i)] = model.fit(new_data.to_numpy(),new_result.to_numpy())
        
    return models

def find_threshold(model,data,column):
    probs = model.predict_proba(data)[:,1]
    
    md = None
    dist = np.inf
       
    for p,i in zip(probs,data[column]):

        if abs(p-0.5) < dist:
            md = i
            dist = abs(p-0.5)

    return md

def get_bounds(UQdata,results,column,binary = False):

    data = pd.DataFrame({c: [i.midpoint() if i.__class__.__name__ == 'Interval' else i for i in UQdata[c]] for c in UQdata.columns},index = UQdata.index) # assume midpoints everywhere else
    
    bounds = {
        'minimum': data.copy(),    
        'maximum': data.copy()
    }


    for i in UQdata.index:
        if UQdata.loc[i,column].__class__.__name__ == "Interval":
            bounds['minimum'].loc[i,column] = UQdata.loc[i,column].Left
            # maximum
            bounds['maximum'].loc[i,column] = UQdata.loc[i,column].Right
    # print(bounds)
    models = {}
    # predict from bounds
    for n, b in bounds.items():

        models[n] = LogisticRegression(max_iter = 1000).fit(b,results.to_numpy())

    minThreshold = find_threshold(models['minimum'],data,column)
    maxThreshold = find_threshold(models['maximum'],data,column)

    bounds2 = {        
        'minTs': data.copy(),
        'maxTs': data.copy(),
        'minTm': data.copy(),
        'maxTm': data.copy()
        }
    
    for i, j in zip(UQdata.index,UQdata[column]):
        if j.__class__.__name__ == "Interval":
            if abs(minThreshold - j.Left) > abs(minThreshold - j.Right):
                bounds2['minTs'].loc[i,column] = j.Left
            else:
                bounds2['minTs'].loc[i,column] = j.Right

            if abs(maxThreshold - j.Left) > abs(maxThreshold - j.Right):
                bounds2['maxTs'].loc[i,column] = j.Left
            else:
                bounds2['maxTs'].loc[i,column] = j.Right

            if j.straddles(maxThreshold):
                bounds2['maxTm'].loc[i,column] = maxThreshold
            elif abs(maxThreshold - j.Left) < abs(maxThreshold - j.Right):
                bounds2['maxTm'].loc[i,column] = j.Left
            else:
                bounds2['maxTm'].loc[i,column] = j.Right

            if j.straddles(minThreshold):
                bounds2['minTm'].loc[i,column] = minThreshold
            elif abs(minThreshold - j.Left) > abs(minThreshold - j.Right):
                bounds2['minTm'].loc[i,column] = j.Left
            else:
                bounds2['minTm'].loc[i,column] = j.Right

    # predict from bounds
    for n, b in bounds2.items():
        # d1 = np.array(b).reshape(-1,1)
        model = LogisticRegression(max_iter = 1000)
        models[n] = model.fit(b,results.to_numpy())

            
    return {str(column)+'_'+k:i for k,i in models.items()}

def int_logistic_regression(UQdata,results):

    uq_col = []
    
    for c in UQdata.columns:
        # check which columns have interval data
        for i in UQdata[c]:
            if i.__class__.__name__ == 'Interval':
                uq_col.append(c)
                break
    
    data = {'lower':pd.DataFrame({
                **{c:[i.Left for i in UQdata[c]] for c in uq_col},
                **{c:UQdata[c] for c in UQdata.columns if c not in uq_col}
                }, index = UQdata.index),
            'upper':pd.DataFrame({
                **{c:[i.Right for i in UQdata[c]] for c in uq_col},
                **{c:UQdata[c] for c in UQdata.columns if c not in uq_col}
                }, index = UQdata.index)
            }

    models = {k:LogisticRegression(max_iter = 1000).fit(d,results) for k,d in data.items()}
        
    return models

def ROC(model = None, predictions = None, data = None, results = None, uq = False, drop = True):
    
    s = []
    fpr = []
    
    if predictions is None:
        predictions = model.predict_proba(data)[:,1]
    
    for p in tqdm(np.linspace(0,1,1001),desc='ROC calcualtion'):
        a = 0
        b = 0
        c = 0
        d = 0

        for prob, result in zip(predictions,results):
            if uq and not drop:
                if (prob[0] >= p) and (prob[1] >= p):
                    x = 1
                elif (prob[0] < p) and (prob[1] < p):
                    x = 0
                else:
                    x = 0.5
                    
                if x == 0.5:
                    if result:
                        a += pba.I(0,1)
                        b += pba.I(0,1)
                    else:
                        c += pba.I(0,1)
                        d += pba.I(0,1)
                elif x:
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
                
                        
            else:
                if uq and drop:
                    if (prob[0] >= p) and (prob[1] >= p):
                        x = 1
                    elif (prob[0] < p) and (prob[1] < p):
                        x = 0
                    else:
                        continue                 
                else:
                    x = prob >= p
                    
                if x:
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
                    
        if a == 0:
            s.append(0)
        else:
            s.append(1/(1+(c/a)))
        
        if b == 0:
            fpr.append(0)
        else:
            fpr.append(1/(1+(d/b)))
        
    return s, fpr
   
def UQ_ROC(models, data, results):
    
    s = []
    fpr = []
    
    predictions = []
    
    for d in data.index:
        l = [m.predict_proba(data.loc[d].to_numpy().reshape(1, -1))[:,1] for k,m in models.items()]
        predictions.append((min(l),max(l)))
    
        
    s_i,fpr_i = ROC(predictions = predictions, data = data, results = results, uq = True, drop = False)
    s_t,fpr_t = ROC(predictions = predictions, data = data, results = results, uq = True, drop = True)
    
    s_i = [pba.I(i) for i in s_i]
    fpr_i = [pba.I(i) for i in fpr_i]
    

    return s_i, fpr_i, s_t, fpr_t


def auc(s,fpr):
    
    fpr,s = zip(*sorted(zip(fpr,s)))
        
    return np.trapz(s,fpr)

def UQ_ROC_alt(models, data, results):
   
    s = []
    fpr = []
    
    Sigma = []
    Tau = []
    Nu = []
    
    
    probabilities = []
    for d in data.index:
        l = [m.predict_proba(data.loc[d].to_numpy().reshape(1, -1))[:,1] for k,m in models.items()]
        probabilities.append((min(l),max(l)))
    
    
    for p in tqdm(np.linspace(0,1,1001),desc = 'UQ_ROC_Calculation'):
        a = 0
        b = 0
        c = 0
        d = 0
        
        sigma = 0
        tau = 0
        nu = 0
        

        for prob, result in zip(probabilities,results):

            if (prob[0] >= p) and (prob[1] >= p):
                x = 1
            elif (prob[0] < p) and (prob[1] < p):
                x = 0
            else:
                x = 0.5
                nu += 1
                
            if x == 0.5:
                if result:
                    sigma += 1
                else:
                    tau += 1
            elif x:
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
                    
        if a == 0:
            s.append(0)
        else:
            s.append(1/(1+(c/a)))
        
        if b == 0:
            fpr.append(0)
        else:
            fpr.append(1/(1+(d/b)))
            
        Sigma.append(sigma/sum(results))
        Tau.append(tau/(len(results) - sum(results)))
        Nu.append(nu/len(results))
            
        
    return s, fpr, Sigma, Tau, Nu
   
def check_int_MC(models,UQdata,results,many,test_data):
    
    int_probabilities = [pba.I(min(i),max(i)) for i in zip(*[m.predict_proba(test_data)[:,1] for m in models.values()])]
    
    ir = 0
    oor = 0
    
    for i in tqdm(range(many),desc = 'Monte Carlo search'):
        new_data = UQdata.copy()
        for j in new_data.index:
            for k in new_data.columns:
                if new_data.loc[j,k].__class__.__name__ == 'Interval':
                    new_data.loc[j,k] = new_data.loc[j,k].Left + random.random()*new_data.loc[j,k].width()
        # print(new_data)
        new_model = LogisticRegression(max_iter = 1000)
        new_model.fit(new_data,results)
        probabilities = new_model.predict_proba(test_data)[:,1]
        for ip, p in zip(int_probabilities,probabilities):
            # print(ip,p)
            if ip.straddles(p,endpoints = True):
                ir += 1
            else:
                oor += 1
                
    return ir, oor
        
    
__all__ = [
    'generate_confusion_matrix',
    'uc_logistic_regression',
    'int_logistic_regression',
    'ROC',
    'UQ_ROC',
    'auc',
    'UQ_ROC_alt',
    'check_int_MC'
]
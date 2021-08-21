import numpy as np
import pandas as pd
from tqdm import tqdm
import pba
from scipy.stats import chi2

def midpoints(data):
    n_data = data.copy()
    for c in data.columns:
        for i in data.index:
            if data.loc[i,c].__class__.__name__ == 'Interval':
                n_data.loc[i,c] = data.loc[i,c].midpoint()
            
    return n_data

def generate_confusion_matrix(results,predictions,throw = False,func = None):
    
    if func is not None:
        predictions = [func(p) for p in predictions]
        
    a = 0
    b = 0
    c = 0
    d = 0
    
    for prediction, result in zip(predictions,results):

        if result:
            if pba.xtimes(prediction):
                a += prediction
                c += prediction
            elif prediction:
                a += 1
            else:
                c += 1
        else:
            if pba.xtimes(prediction):
                b += prediction
                d += prediction
            elif prediction:
                b += 1
            else:
                d += 1

    if throw:
        
        if a.__class__.__name__ != 'Interval':
            a = pba.I(a)
            c = pba.I(c)
        if b.__class__.__name__ != 'Interval':
            b = pba.I(b)     
            d = pba.I(d)
                   
        e = a.width()
        f = b.width()
        
        return a.left,b.left,c.left,d.left,e,f
    else:
        return a,b,c,d    
      
def ROC(model, data, results, func = (lambda x: x)):
    
    s = []
    fpr = []

    total_positive = sum(results)
    total_negative = len(results) - total_positive
    
    probabilities = model.predict_proba(data)[:,1]
    
    for p in tqdm(np.linspace(0,1,1000),desc = 'ROC'):
        
        predictions = [func(prob >= p) for prob in probabilities]
        
        a,b,c,d = generate_confusion_matrix(predictions,results)
        
        s.append(a/total_positive)
        fpr.append(b/total_negative)
        
    return s, fpr, probabilities

def auc(s,fpr):
    
    fpr,s = zip(*sorted(zip(fpr,s)))
        
    return np.trapz(s,fpr)

def incert_ROC(models, data, results):
   
    s = []
    fpr = []
    
    Sigma = []
    Tau = []

    total_positive = sum(results)
    total_negative = len(results) - total_positive
    
    probabilities = models.predict_proba(data)[:,1]
    
    for p in tqdm(np.linspace(0,1,1000),desc = 'ROC'):
        
        predictions = [prob >= p for prob in probabilities]
        
        a,b,c,d,e,f = generate_confusion_matrix(predictions,results, throw = True)
        
        s.append(a/(total_positive-e))
        fpr.append(b/(total_negative-f))
        
        Sigma.append(e/total_positive)
        Tau.append(f/total_negative)
        
    return s, fpr, Sigma, Tau

def hosmer_lemeshow_test(model, data ,results,g = 10):
    
    probs = pd.DataFrame(model.predict_proba(data),index = data.index)
    probs.sort_values(by = 1,inplace = True)
 
    # number of datapoints
    N = len(data.index)
    # number of items in buckets
    s = N/g
    
    buckets = {}
    
    for i in range(g):
        
        idx = probs.iloc[round(s*i):round(s*(1+i))].index
        
        buckets[i] = {
            'observed_cases': sum(results.loc[idx]),
            'expected_cases': sum(probs.loc[idx,1]),
            'observed_n_cases': len(idx) - sum(results.loc[idx]),
            'expected_n_cases': sum(probs.loc[idx,0])
        }
        
    buckets = pd.DataFrame(buckets).transpose()
    buckets.to_csv('b.csv')
    hl = sum(((buckets['observed_cases']-buckets['expected_cases'])**2)/(buckets['expected_cases'])) + sum(((buckets['observed_n_cases']-buckets['expected_n_cases'])**2)/(buckets['expected_n_cases']))

    pval = 1-chi2.cdf(hl,g-2)
    
    return hl, pval
  
def UQ_hosmer_lemeshow_test(models, data, results, g=10):

    probabs_0 = []
    probabs_1 = []
    probabs_mp = []
    
    for d in tqdm(data.index, desc = 'UQ HL Test'):
        p = [(m.predict_proba(data.loc[d].to_numpy().reshape(1, -1))[:,0],m.predict_proba(data.loc[d].to_numpy().reshape(1, -1))[:,1]) for m in models]
        
        p = list(zip(*p))
        
        probabs_0.append(pba.I(min(p[0]),max(p[0])))
        probabs_1.append(pba.I(min(p[1]),max(p[1])))
        probabs_mp.append(np.mean(p[1]))
        
    
    probs = pd.DataFrame({0:probabs_0,1:probabs_1,'mp':probabs_mp},index = data.index, dtype = 'O')
    probs.sort_values(by = 'mp',inplace = True)

    # number of datapoints
    N = len(data.index)
    # number of items in buckets
    s = N/g
    
    buckets = {}
    
    for i in range(g):
        
        idx = probs.iloc[round(s*i):round(s*(1+i))].index

        buckets[i] = {
            'observed_cases': pba.I(sum(results.loc[idx])),
            'expected_cases': sum(probs.loc[idx,1]),
            'observed_n_cases': pba.I(len(idx) - sum(results.loc[idx])),
            'expected_n_cases': sum(probs.loc[idx,0])
        }
    
    pd.DataFrame().from_dict(buckets, orient='index').to_csv('a.csv')
    hl = 0
    for i in range(g):
        
        cl = ((buckets[i]['observed_cases'].left - buckets[i]['expected_cases'].left)**2)/buckets[i]['expected_cases'].left
        cr = ((buckets[i]['observed_cases'].right - buckets[i]['expected_cases'].right)**2)/buckets[i]['expected_cases'].right
        
        nl = ((buckets[i]['observed_n_cases'].left - buckets[i]['expected_n_cases'].left)**2)/buckets[i]['expected_n_cases'].left
        nr = ((buckets[i]['observed_n_cases'].right - buckets[i]['expected_n_cases'].right)**2)/buckets[i]['expected_n_cases'].right
            
        if (buckets[i]['observed_cases'] - buckets[i]['expected_cases']).straddles_zero(endpoints = True):
            hl += pba.I(0, pba.I(cl + nr, cr + nl))
        else:
            hl += pba.I(cl + nr, cr + nl)

    pval = pba.I(1-chi2.cdf(hl.left,g-2),1-chi2.cdf(hl.right,g-2))
    
    return hl, pval

__all__ = [
    'generate_confusion_matrix',
    'ROC',
    'incert_ROC',
    'auc',
    'hosmer_lemeshow_test',
    'UQ_hosmer_lemeshow_test'
    
]
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

def generate_confusion_matrix(results,predictions,throw = False):

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
        e = a.width()
        f = b.width()
        
        return a.left,b.left,c.left,d.left,e,f
    else:
        return a,b,c,d    
    

def ROC(model = None, predictions = None, data = None, results = None, uq = False, drop = True):

    s = []
    fpr = []
    
    if predictions is None:
        predictions = model.predict_proba(data)[:,1]
    
    for p in tqdm(np.linspace(0,1,1000),desc='ROC calcualtion'):
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
        
    return s, fpr, predictions
   
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

def UQ_ROC_alt(models, data, results):
   
    s = []
    fpr = []
    
    Sigma = []
    Tau = []
    
    probabilities = models.predict_proba(data)[:,1]

    for p in tqdm(np.linspace(0,1,1000),desc = 'UQ ROC'):
        a = 0
        b = 0
        c = 0
        d = 0
        
        sigma = 0
        tau = 0
        nu = 0
        
        for prob, result in zip(probabilities,results):

            if pba.always(prob >= p):
                if result:
                    # true positive
                    a += 1
                else:
                    # false positive
                    b+= 1
            elif pba.always(prob < p):
                if result:
                    # false negative
                    c += 1
                else:
                    # true negative
                    d += 1
            else:
                nu += 1

                if result:
                    sigma += 1
                else:
                    tau += 1
              
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
    hl = sum(((buckets['observed_cases']-buckets['expected_cases'])**2)/(buckets['expected_cases'])) + sum(((buckets['observed_n_cases']-buckets['expected_n_cases'])**2)/(buckets['expected_n_cases']))

    pval = 1-chi2.cdf(hl,g-2)
    
    return hl, pval
  
def UQ_hosmer_lemeshow_test(models, data, results, g=10):
    
    probabs_0 = []
    probabs_1 = []
    probabs_mp = []
    
    for d in tqdm(data.index, desc = 'UQ HL Test'):
        p = [(m.predict_proba(data.loc[d].to_numpy().reshape(1, -1))[:,0],m.predict_proba(data.loc[d].to_numpy().reshape(1, -1))[:,1]) for k,m in models.items()]
        
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
        
    hl = 0
    for i in range(g):
        
        cl = ((buckets[i]['observed_cases'].left - buckets[i]['expected_cases'].left)**2)/buckets[i]['expected_cases'].left
        cr = ((buckets[i]['observed_cases'].Right - buckets[i]['expected_cases'].Right)**2)/buckets[i]['expected_cases'].Right
        
        nl = ((buckets[i]['observed_n_cases'].left - buckets[i]['expected_n_cases'].left)**2)/buckets[i]['expected_n_cases'].left
        nr = ((buckets[i]['observed_n_cases'].Right - buckets[i]['expected_n_cases'].Right)**2)/buckets[i]['expected_n_cases'].Right
            
        if pba.always(buckets[i]['observed_cases'] - buckets[i]['expected_cases'] != 0):
            hl += pba.I(cl+nr,cr+nl)
        else:
            hl += pba.I(0,pba.I(cl+nr,cr+nl))
            
            

    pval = pba.I(1-chi2.cdf(hl.left,g-2),1-chi2.cdf(hl.Right,g-2))
    
    return hl, pval

__all__ = [
    'generate_confusion_matrix',
    'ROC',
    'auc',
    'UQ_ROC_alt',
    'hosmer_lemeshow_test',
    'UQ_hosmer_lemeshow_test'
    
]
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import itertools as it
from tqdm import tqdm
import pba
import random
import scipy.optimize as so
from scipy.stats import chi2
from scipy.stats import chisquare

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
  
        if prediction.__class__.__name__ != 'Interval':
            prediction = pba.I(prediction,prediction)

        if prediction.left == prediction.right:
            if result:
                if prediction.left:
                    a += 1
                else:
                    c += 1
            else:
                if prediction.left:
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

def uc_logistic_regression(data,result,uncertain,nested = False):
    
    models = {}
    
    intercepts = [np.inf,-np.inf]
    coefs = [[np.inf,-np.inf]*len(data.columns)]
    
    
    for N,i in tqdm(enumerate(it.product([0,1],repeat=len(uncertain))),total=2**len(uncertain),desc='UC Logistic Regression',leave=(not nested)):
        keep = False
        new_data = pd.concat((data,uncertain), ignore_index = True)
        new_result = pd.concat((result, pd.Series(i)), ignore_index = True)

        model = LogisticRegression(max_iter=1000)       
        model.fit(new_data.to_numpy(),new_result.to_numpy())

        if model.intercept_[0] < intercepts[0]:
            intercepts[0] = model.intercept_
            keep = True
        if model.intercept_[0] > intercepts[1]:
            intercepts[1] = model.intercept_    
            keep = True   
            
        for ii, (c,c_) in enumerate(zip(coefs,model.coef_[0])):
            if c_ < c[0]:
                coefs[ii][0] = c_
                keep = True
            if c_ > c[1]:
                coefs[ii][1] = c_
                keep = True
                   
        if keep:  
            models[str(i)] = model.fit(new_data.to_numpy(),new_result.to_numpy())
        
    return models
   
def find_thresholds(B0,B,UQdata,uq_cols,binary_cols = []):

    def F(B0,B,X):
        f = float(B0)
        for b,x in zip(B,X):
            f += float(b*x)
        return f

    def min_F(X,Bx,Cx):
        for x,Bx in zip(X,Bx):
            Cx += float(b*x)
        return abs(Cx)
    
    def max_F(X,Bx,Cx):
        for x,Bx in zip(X,Bx):
            Cx += float(b*x)
        return -abs(Cx)        
        
        
    left = lambda x: x.left
    right = lambda x: x.Right
    
    dataMin = UQdata.copy()
    dataMax = UQdata.copy()
    
    # need to find the min/max spread around these points

    for j in UQdata.index:

        X = UQdata.loc[j].copy()
        Xmin = X.copy()
        Xmax = X.copy()
            
        Cx = float(B0)
        Bx = []
        X0 = []
        uncertain_cols = []
        
        for i,b in zip(X.index,B):

            if i in uq_cols and X[i].__class__.__name__ == 'Interval':
                Bx.append(b)
                X0.append(X[i].midpoint())
                uncertain_cols.append(i)
            else:
                Cx += float(float(X[i])*b)

        
        if len(uncertain_cols) != 0:

            bounds = [(X[i].left,X[i].Right) for i in uq_cols if X[i].__class__.__name__ == 'Interval']

            Rmin = so.minimize(min_F,X0,args = (Bx,Cx),method = 'L-BFGS-B',bounds = bounds)
            Rmax = so.minimize(max_F,X0,args = (Bx,Cx),method = 'L-BFGS-B',bounds = bounds)

            for i,xmin,xmax in zip(uncertain_cols,Rmin.x,Rmax.x):
                if i in binary_cols:
                    if xmin not in (0,1):
                        xmin = round(xmin)
                        
                Xmin[i] = xmax
                Xmax[i] = xmin


        dataMin.loc[j] = Xmin
        dataMax.loc[j] = Xmax


    return dataMin, dataMax

def int_logistic_regression(UQdata,results,binary_cols = []):

    left = lambda x: x.left
    right = lambda x: x.Right
    
    uq_col = binary_cols.copy()
    
    for c in UQdata.columns:
        
        if c in binary_cols:
            continue
        
        # check which columns have interval data       
        for i in UQdata[c]:
            if i.__class__.__name__ == 'Interval':
                uq_col.append(c)
                break
    
    
    data = {''.join(k):pd.DataFrame({
                **{c:[F(i) if i.__class__.__name__ == 'Interval' else i for i in UQdata[c]] for c,F in zip(uq_col,func)},
                **{c:UQdata[c] for c in UQdata.columns if c not in uq_col}
                }, index = UQdata.index).reindex(columns = UQdata.columns)
             for k, func in tqdm(zip(it.product('lr',repeat = len(uq_col)),it.product((left,right),repeat = len(uq_col))),desc='Getting Bounds (1)',total = 2**len(uq_col))
            }
    models = {k:LogisticRegression(max_iter = 1000).fit(d,results) for k,d in tqdm(data.items(),desc ='Fitting Models (1)')}

    n_data = {}
    for k,m in tqdm(models.items(),desc='Getting Bounds (2)'):
        B0 = m.intercept_
        B = m.coef_[0]

        nMin, nMax = find_thresholds(B0,B,UQdata,uq_col,binary_cols = binary_cols)

        n_data[k+'min'] = nMin.reindex(columns = UQdata.columns)
        n_data[k+'max'] = nMax.reindex(columns = UQdata.columns)


    n_models = {k:LogisticRegression(max_iter = 1000).fit(d,results) for k,d in tqdm(n_data.items(),desc ='Fitting Models (2)')}

    return {**models,**n_models}

def uc_int_logistic_regression(UQdata,results,uncertain,binary_cols = [],uc_index=[]):

    left = lambda x: x.left
    right = lambda x: x.Right
   
    uq_col = binary_cols.copy()
    
    for c in UQdata.columns:
        
        if c in binary_cols:
            continue
        
        # check which columns have interval data       
        for i in UQdata[c]:
            if i.__class__.__name__ == 'Interval':
                uq_col.append(c)
                break
    
    
    # remove uc classification from dataset when training model
    kc_index = [i for i in UQdata.index if i not in uc_index]
    
    data1 = {''.join(k):pd.DataFrame({
                **{c:[F(i) if i.__class__.__name__ == 'Interval' else i for i in UQdata[c]] for c,F in zip(uq_col,func)},
                **{c:UQdata[c] for c in UQdata.columns if c not in uq_col}
                }, index = UQdata.index).reindex(columns = UQdata.columns)
             for k, func in tqdm(zip(it.product('lr',repeat = len(uq_col)),it.product((left,right),repeat = len(uq_col))),desc='Getting Bounds (1)',total = 2**len(uq_col))
            }
    
    int_models = {k:LogisticRegression(max_iter = 1000).fit(d.loc[kc_index],results.loc[kc_index]) for k,d in tqdm(data1.items(),desc ='Fitting Models (1)')}

    n_data = {}
    for k,m in tqdm(int_models.items(),desc='Getting Bounds (2)'):
        B0 = m.intercept_
        B = m.coef_[0]

        nMin, nMax = find_thresholds(B0,B,UQdata,uq_col,binary_cols = binary_cols)

        n_data[k+'min'] = nMin.reindex(columns = UQdata.columns)
        n_data[k+'max'] = nMax.reindex(columns = UQdata.columns)

    uc_models = {}
    for ii, dataset in tqdm({**data1,**n_data}.items(),desc='datasets'):

        for jj, model in uc_logistic_regression(dataset.loc[kc_index],results.loc[kc_index],dataset.loc[uc_index],nested=True).items():
            uc_models[ii+jj] = model
    
    return uc_models

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
   
def UQ_ROC(model, data, results, func = (lambda x: x)):
    
    s = []
    fpr = []

    total_positive = sum(results)
    total_negative = len(results) - total_positive
    
    probabilities = model.predict_proba(data)[:,1]
    
    for p in tqdm(np.linspace(0,1,1000),desc = 'ROC'):
        
        prediction = [func(prob >= p) for prob in probabilities]
        
        a = b = c = d = 0
        
        for i, j in zip(prediction,results):
            if isinstance(i, pba.Logical):
                if j:
                    if pba.xtimes(i):
                        a += i
                        c += i
                    elif i:
                        a += 1
                    else:
                        b += 1
                else:
                    if pba.xtimes(i):
                        b += i
                        d += i
                    elif i:
                        b += 1
                    else:
                        d += 1
        
        if isinstance(a,pba.Interval):
            s += [a/total_positive]
            fpr += [b/total_negative]
        
    return s, fpr, probabilities

def auc(s,fpr):
    
    fpr,s = zip(*sorted(zip(fpr,s)))
        
    return np.trapz(s,fpr)

def UQ_ROC_alt(models, data, results):
   
    s = []
    fpr = []
    
    Sigma = []
    Tau = []
    Nu = []
    
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
                    new_data.loc[j,k] = new_data.loc[j,k].left + random.random()*new_data.loc[j,k].width()
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
        
    # print(buckets)
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
    'uc_logistic_regression',
    'int_logistic_regression',
    'uc_int_logistic_regression',
    'ROC',
    'UQ_ROC',
    'auc',
    'UQ_ROC_alt',
    'check_int_MC',
    'hosmer_lemeshow_test',
    'UQ_hosmer_lemeshow_test'
    
]
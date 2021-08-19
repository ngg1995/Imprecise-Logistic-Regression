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

class ImpLogReg:
    
    def __init__(self):
        self.models = {}
        
    def fit(self, data, results, uncertain_classification: bool, interval_dataset: bool, uncertain_data: pd.DataFrame = None, binary_cols = [], uc_index=[]):
        
        assert uncertain_classification is not False and interval_dataset is not False
        
        if uncertain_classification and interval_dataset:
            self.models = _uc_int(data,results,uncertain_data,binary_cols,uc_index)
        
        elif uncertain_classification:
            self.models = _uc(data, results, uncertain_data)
            
        elif interval_dataset:
            self.models = _int_data(data, results, binary_cols)

    
def _uc(data,result,uncertain,nested = False):
    
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

def _int_data(UQdata,results,binary_cols = []):

    left = lambda x: x.Left
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

        nMin, nMax = _find_thresholds(B0,B,UQdata,uq_col,binary_cols = binary_cols)

        n_data[k+'min'] = nMin.reindex(columns = UQdata.columns)
        n_data[k+'max'] = nMax.reindex(columns = UQdata.columns)


    n_models = {k:LogisticRegression(max_iter = 1000).fit(d,results) for k,d in tqdm(n_data.items(),desc ='Fitting Models (2)')}

    return {**models,**n_models}

def _uc_int(UQdata, results, binary_cols = [], uc_index=[]):

    left = lambda x: x.left
    right = lambda x: x.right
   
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

        nMin, nMax = _find_thresholds(B0,B,UQdata,uq_col,binary_cols = binary_cols)

        n_data[k+'min'] = nMin.reindex(columns = UQdata.columns)
        n_data[k+'max'] = nMax.reindex(columns = UQdata.columns)

    uc_models = {}
    for ii, dataset in tqdm({**data1,**n_data}.items(),desc='datasets'):

        for jj, model in uc(dataset.loc[kc_index],results.loc[kc_index],dataset.loc[uc_index],nested=True).items():
            uc_models[ii+jj] = model
    
    return uc_models

def _find_thresholds(B0,B,UQdata,uq_cols,binary_cols = []):

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
        
        
    left = lambda x: x.Left
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

            bounds = [(X[i].Left,X[i].Right) for i in uq_cols if X[i].__class__.__name__ == 'Interval']

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

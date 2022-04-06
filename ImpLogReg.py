from operator import mod
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import itertools as it
from tqdm import tqdm
import pba
import scipy.optimize as so

class ImpLogReg:
    
    def __init__(self, uncertain_data = False, uncertain_class = False, **kwargs):

        self.models = {}
        self.uncertain_data = uncertain_data
        self.uncertain_class = uncertain_class
        
        self.params = LogisticRegression(**kwargs).get_params()
        
        self.__dict__.update((k, v) for k,v in self.params.items())
        self.coef_ = []
        self.intercept_ = []
        self.n_iter_ = []
        self.classes_ = None
        self.n_features_in_ = None
    
    def __iter__(self):
        for _, m in self.models.items():
            yield m
    
    def __len__(self):
        return len(self.models.keys())

    def decision_function(self,X):
        
        decision = np.empty((len(self),len(X)))
        
        for i, model in enumerate(self):
            decision[i] = model.predict(X)
        
        return [pba.I(i) for i in decision.T]
    
    def densify(self):
        assert len(self) > 0
        for k,m in self.models.items():
            self.models[k] = m.densify()
            
    def fit(self, data, results ,sample_weight=None, catagorical = []):
        
        self.params.update({k:v for k,v in self.__dict__.items() if k in self.params.keys()})
        
        if self.uncertain_class:
            # need to strip the unlabled data from the data set
            unlabeled_data = data.copy()
            labeled_data = data.copy()
            labeled_results = results.copy()
            
            for i in results.index:

                if pba.always(results.loc[i]==0) or pba.always(results.loc[i]==1):
                    unlabeled_data.drop(i, axis = 'index', inplace = True)
                else:
                    labeled_data.drop(i, axis = 'index', inplace = True)
                    labeled_results.drop(i, axis = 'index', inplace = True)
                               
            labeled_results = labeled_results.convert_dtypes()
            
            if self.uncertain_data:
                self.models = _uc_int(labeled_data, labeled_results, unlabeled_data, sample_weight, catagorical, self.params)
            else:
                self.models = _uncertain_class(labeled_data, labeled_results, unlabeled_data, sample_weight = sample_weight, nested = False, params = self.params)
            
        elif self.uncertain_data:
            self.models = _int_data(data,results,sample_weight,catagorical,self.params)
            
        else:
            self.models = {0: LogisticRegression(**self.params).fit(data, results, sample_weight)}
            
        for i, m in enumerate(self):
            if i == 0:
                self.classes_ = m.classes_
                self.n_features_in_ = m.n_features_in_
                
            self.coef_.append(m.coef_)
            self.intercept_.append(m.intercept_)
            self.n_iter_ = np.array(m.n_iter_)
        
    def get_params(self):
        return self.params
    
    def predict(self, X):
        
        predictions = np.empty((len(self),len(X)))
        
        for i, model in enumerate(self):
            predictions[i] = model.predict(X)
        
        return [pba.I(i).to_logical() for i in predictions.T]
    
    def predict_proba(self, X):
        
        predictions0 = np.empty((len(self),len(X)))
        predictions1 = np.empty((len(self),len(X)))
        
        
        for i, model in enumerate(self):
            predictions0[i] = model.predict_proba(X)[:,0]
            predictions1[i] = model.predict_proba(X)[:,1]
        
        return np.array([[pba.I(i) for i in predictions0.T],[pba.I(i) for i in predictions1.T]]).T
    
    def predict_log_proba(self, X):
        
        predictions = np.empty((len(self),len(X)))
        
        for i, model in enumerate(self):
            predictions[i] = model.predict_log_proba(X)[:,0]
        
        return [pba.I(i) for i in predictions.T]
    
    def score(self,X,y, sample_weight = None):
        
        scores = [model.score(X,y,sample_weight) for model in self]
        return pba.I(scores)
    
    def set_params(self, **kwargs):
        for param, val in kwargs.items():
            try:
                self.params[param] = val
            except:
                print("WARNING: Parameter %s not found" %(param))
        self.__dict__.update((k, v) for k,v in self.params)
        
                

    def sparsify(self):
        assert len(self) > 0
        for k,m in self.models.items():
            self.models[k] = m.sparsify()
                      
def _uncertain_class(data: pd.DataFrame, result: pd.Series, uncertain: pd.DataFrame, sample_weight = None, nested = False, params = {}) -> dict:
    
    models = {}

    for N,i in tqdm(enumerate(it.product([0,1],repeat=len(uncertain))),total=2**len(uncertain),desc='UC Logistic Regression',leave=(not nested)):

        new_data = pd.concat((data,uncertain), ignore_index = True)
        new_result = pd.concat((result, pd.Series(i)), ignore_index = True).convert_dtypes()
        
        model = LogisticRegression(**params)       
        model.fit(new_data.to_numpy(),new_result.to_numpy(dtype = bool),sample_weight=sample_weight)

        models[str(i)] = model
        
    return models

def _int_data(data,results,sample_weight,catagorical,params, nested = False) -> dict:
    
    uq = [(i,c) for i in data.index for c in data.columns if data.loc[i,c].__class__.__name__ == 'Interval']
                    
    
    def get_vals_from_intervals(r,data,uq,cat=[]):

        ndata = data.copy()
        
        for (i,c), v in zip(uq,r):
            if c in cat:
                ndata.loc[i,c] = np.round(v)
            else:
                ndata.loc[i,c] = ndata.loc[i,c].left + v*ndata.loc[i,c].width()
                    
        return ndata


    def find_bounds(r, data, results, uq, params,ci, mm, n = 0, cat = []) -> float:
        lr = LogisticRegression(**params).fit(get_vals_from_intervals(r,data, uq,cat),results)

        if ci == 'intercept':
            if mm == 'min':
                return  lr.intercept_[0]
            else:
                return  -lr.intercept_[0]
        else:
            if mm == 'min':
                return  lr.coef_[:,n]
            else:
                return  -lr.coef_[:,n]
          
    def find_xlr_model(data, results, uq, params,s,b ,ci, mm, n = 0, cat = []):
        method = 'Nelder-Mead'
        bounds = so.minimize(find_bounds, np.ones(s), args = (data,results, uq,params, ci, mm),bounds = b, method=method)
        return LogisticRegression(**params).fit(get_vals_from_intervals(bounds.x,data, uq),results)
        
    s = len(uq)
    b = s*[(0,1)]
    t = tqdm(total = 2+2*len(data.columns))
    t.update(2)

    models = {
        'leftmost': LogisticRegression(**params).fit(get_vals_from_intervals([0]*s,data,uq),results),
        'rightmost': LogisticRegression(**params).fit(get_vals_from_intervals([1]*s,data,uq),results),
    }
    
    for mm in ['min','max']:
        models[f'r_{mm}_intercept'] = find_xlr_model(data, results, uq, params,s,b, 'intercept', mm)
        t.update()
        for i,c in enumerate(data.columns):
            models[f'r_{mm}_coef_{i}'] = find_xlr_model(data, results, uq, params,s,b, 'coef',mm,i,catagorical)
            t.update()
        
    return models

def _uc_int(data, results, uncertain, sample_weight, catagorical, params) -> dict:
    
    models = {}

    for N,i in tqdm(enumerate(it.product([0,1],repeat=len(uncertain))),total=2**len(uncertain),desc='UC Logistic Regression'):

        new_data = pd.concat((data,uncertain), ignore_index = True)
        new_result = pd.concat((results, pd.Series(i)), ignore_index = True).convert_dtypes()
        
        models = {
            **models,
            **{"%s_%s"%(i,k):v for k,v in _int_data(new_data,new_result,sample_weight,catagorical,params, nested = True).items()}
            }
    
    return models

def _find_thresholds(B0,B,data,uq_cols,catagorical):

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
    right = lambda x: x.right
    
    dataMin = data.copy()
    dataMax = data.copy()
    
    # need to find the min/max spread around these points

    for j in data.index:

        X = data.loc[j].copy()
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

            bounds = [(X[i].left,X[i].right) for i in uq_cols if X[i].__class__.__name__ == 'Interval']

            Rmin = so.minimize(min_F,X0,args = (Bx,Cx),method = 'L-BFGS-B',bounds = bounds)
            Rmax = so.minimize(max_F,X0,args = (Bx,Cx),method = 'L-BFGS-B',bounds = bounds)

            for i,xmin,xmax in zip(uncertain_cols,Rmin.x,Rmax.x):
                if i in catagorical:
                    if xmin not in (0,1):
                        xmin = round(xmin)
                        
                Xmin[i] = xmax
                Xmax[i] = xmin


        dataMin.loc[j] = Xmin
        dataMax.loc[j] = Xmax


    return dataMin, dataMax

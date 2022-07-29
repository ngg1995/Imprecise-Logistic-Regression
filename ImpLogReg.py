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
        
        if self.uncertain_class and self.uncertain_data:
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
            

            self.models = _uc_int(labeled_data, labeled_results, unlabeled_data, sample_weight, catagorical, self.params)

        elif self.uncertain_class:
            # want to get uq_data_index
            uq_data_index = []
    
            for i in results.index:
                if not (pba.always(results.loc[i]==0) or pba.always(results.loc[i]==1)):
                    uq_data_index.append(i)
                    
            self.models = _uncertain_class(data, results,uq_data_index, sample_weight, catagorical, self.params)
            
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

def _uncertain_class(data: pd.DataFrame, result: pd.Series, uq_data_index: list, sample_weight = None, nested = False, params = {}) -> dict:
    
    models = {}

    new_results = result.copy().astype(bool)
    new_results.loc[uq_data_index] = [False]*len(uq_data_index)
    models['zero_most'] = LogisticRegression().fit(data,new_results)

    new_results = result.copy().astype(bool)
    new_results.loc[uq_data_index] = [True]*len(uq_data_index)
    models['one_most'] = LogisticRegression().fit(data,new_results)

    k_0 = lambda x:  abs(models['zero_most'].predict_proba(np.array([x]).reshape(1,-1))[0][1] - 0.5)
    k_1 = lambda x:  abs(models['one_most'].predict_proba(np.array([x]).reshape(1,-1))[0][1] - 0.5)

    k = pba.I(so.minimize(k_1,np.median(data)).x,so.minimize(k_0,np.median(data)).x)

    zones = {
        'left': [],
        'center': [],
        'right': []
    }

    for u in uq_data_index:
        if data.loc[u,0] < k.left:
            zones['left'].append(u)
        elif data.loc[u,0] > k.right:
            zones['right'].append(u)
        else:
            zones['center'].append(u)
            
    t = tqdm(total = 2**(len(zones['center'])+1))
    for l_bool,r_bool in [(True, False),(False,True)]:
        for c_bool in it.product([False,True],repeat=len(zones['center'])):
            new_results = result.copy().astype(bool)
            new_results.loc[zones['left']] = l_bool
            new_results.loc[zones['center']] = c_bool
            new_results.loc[zones['right']] = r_bool
            
            models[str(new_results.loc[uq_data_index].to_list())] = LogisticRegression().fit(data,new_results.astype(bool))
            t.update()
            
    return models

def _int_data(data,results,sample_weight,catagorical,params, nested = False) -> dict:
    
    uq = [(i,c) for i in data.index for c in data.columns if data.loc[i,c].__class__.__name__ == 'Interval']
    
    assert len(uq) != 0
        
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
        bounds = so.minimize(find_bounds, 0.5*np.ones(s), args = (data,results, uq,params, ci, mm),bounds = b, method=method)
        return LogisticRegression(**params).fit(get_vals_from_intervals(bounds.x,data, uq),results), get_vals_from_intervals(bounds.x,data, uq)
        
    s = len(uq)
    n = len(data.columns)
    m = len({c for _,c in uq})
    b = s*[(0,1)]
    t = tqdm(total = 2**m+2*n+2)
    # t.update(2)
        
    models = {}
    dataset = {}
    for i in it.product([0,1],repeat=n):
        x = {c:v for c,v in zip(data.columns,i)}
        r = [x[c] for _,c in uq]
        models[str(i)] = LogisticRegression(**params).fit(get_vals_from_intervals(r,data,uq),results)
        dataset[str(i)] = get_vals_from_intervals(r,data,uq)
        t.update()
                        
        
    for mm in ['min','max']:
        models[f'r_{mm}_intercept'], dataset[f'r_{mm}_intercept'] = find_xlr_model(data, results, uq, params,s,b, 'intercept', mm)
        t.update()
        for i,c in enumerate(data.columns):
            models[f'r_{mm}_coef_{i}'], dataset[f'r_{mm}_coef_{i}'] = find_xlr_model(data, results, uq, params,s,b, 'coef',mm,i,catagorical)
            t.update()

    return models

def _uc_int(data, results, uncertain, sample_weight, catagorical, params) -> dict:
    
    new_data = pd.concat((data,uncertain), ignore_index = True)
    uq = [(i,c) for i in new_data.index for c in new_data.columns if new_data.loc[i,c].__class__.__name__ == 'Interval']
    
    
    def get_vals_from_intervals(r, data, results, uq, catagorical):

        # split r
        i = r[0:len(uq)]
        j = r[len(uq):]
        
        ndata = data.copy()
        
        for (i,c), v in zip(uq,i):
            if c in catagorical:
                ndata.loc[i,c] = np.round(v)
            else:
                ndata.loc[i,c] = ndata.loc[i,c].left + v*ndata.loc[i,c].width()
        
        nresults = pd.concat((results, pd.Series(np.round(j))), ignore_index = True).convert_dtypes()

        return ndata.to_numpy(), nresults.to_numpy(dtype = bool)

    def find_bounds(r, data, results, uq, params,ci, mm, n = 0, cat = []) -> float:
        lr = LogisticRegression(**params).fit(*get_vals_from_intervals(r,data, results, uq, cat))

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
          
    def find_xlr_model(init, data, results, uq, params,s,b ,ci, mm, n = 0, cat = []):
        method = 'Nelder-Mead'
        bounds = so.minimize(find_bounds, init, args = (data,results, uq,params, ci, mm),bounds = b, method=method)
        return LogisticRegression(**params).fit(*get_vals_from_intervals(bounds.x,data, results, uq, cat))
    
    models = {}
    s = len(uq) + len(uncertain)
    b = s*[(0,1)]
    t = tqdm(total = 3*2*2*len(data.columns))
    for j in [0,.5,1]:
        init = j*np.ones(s)
        for mm in ['min','max']:
            models[f'r_{mm}_intercept({j})'] = find_xlr_model(init, new_data, results, uq, params,s,b, 'intercept', mm)
            t.update()
            for i,c in enumerate(data.columns):
                models[f'r_{mm}_coef_{i}({j})'] = find_xlr_model(init, new_data, results, uq, params,s,b, 'coef',mm,i,catagorical)
                t.update()
        
    return models
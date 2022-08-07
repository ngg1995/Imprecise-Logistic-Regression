import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import itertools as it
from tqdm import tqdm
import pba
import scipy.optimize as so

class ImpLogReg:
    
    def __init__(self, 
                 uncertain_data: bool = False, 
                 uncertain_class: bool = False, 
                 **kwargs):

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
            
    def fit(self, data, results, sample_weight=None, catagorical = []):
        
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

    left = lambda x: x.left
    right = lambda x: x.right
    
    uq_col = catagorical.copy()
    
    uq = [(i,c) for i in data.index for c in data.columns if data.loc[i,c].__class__.__name__ == 'Interval']
    
    uq_col = {c for _,c in uq}
    
    data_ = {''.join(k):pd.DataFrame({
                **{c:[F(i) if i.__class__.__name__ == 'Interval' else i for i in data[c]] for c,F in zip(uq_col,func)},
                **{c:data[c] for c in data.columns if c not in uq_col}
                }, index = data.index).reindex(columns = data.columns)
             for k, func in zip(it.product('lr',repeat = len(uq_col)),it.product((left,right),repeat = len(uq_col)))
            }
    models = {}
    # pbar = tqdm(total = 3*2**len(uq_col),leave = True, colour='red',desc='Uncertain Data',position=0)
    for k,d in tqdm(data_.items(),leave = True, colour='red',desc='Uncertain Data (1)',position=0):
        models.update({k:LogisticRegression(**params).fit(d,results.to_numpy(dtype = bool),sample_weight)})


   
    n_models = models.copy()
    for k,m in tqdm(models.items(),leave = True, colour='red',desc='Uncertain Data (2)',position=0):
        for p in tqdm(np.arange(0.01,1,0.01),colour='green', position=1):
            min_data = data.copy()
            max_data = data.copy()
            x0 = [pba.I(np.median(data[c])).midpoint() for c in data.columns]
            x = {c:v for v,c in zip(so.minimize(lambda x:  abs(m.predict_proba(np.array([x]).reshape(1,-1))[0][1] - p),x0).x,data.columns)}
            
            for i,c in uq:
                if data.loc[i,c].straddles(x[c]):
                    min_data.loc[i,c] = x[c]
                    if abs(data.loc[i,c].right - x[c]) > (data.loc[i,c].left - x[c]):
                        max_data.loc[i,c] = data.loc[i,c].right
                        # min_data.loc[i,c] = data.loc[i,c].left
                    else:
                        max_data.loc[i,c] = data.loc[i,c].left
                        # min_data.loc[i,c] = data.loc[i,c].right
                        
                elif pba.always(data.loc[i,c] < x[c]):
                    max_data.loc[i,c] = data.loc[i,c].left
                    min_data.loc[i,c] = data.loc[i,c].right
                else:
                    max_data.loc[i,c] = data.loc[i,c].right
                    min_data.loc[i,c] = data.loc[i,c].left
            
            n_models[f'{k}-min({p})'] = LogisticRegression(**params).fit(min_data,results.to_numpy(dtype = bool),sample_weight)

            n_models[f'{k}-max({p})'] = LogisticRegression(**params).fit(max_data,results.to_numpy(dtype = bool),sample_weight)
            
    return n_models

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


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
    
    for c in data.columns:
        
        if c in catagorical:
            continue
        
        # check which columns have interval data       
        for i in data[c]:
            if i.__class__.__name__ == 'Interval':
                uq_col.append(c)
                break
    
    data_ = {''.join(k):pd.DataFrame({
                **{c:[F(i) if i.__class__.__name__ == 'Interval' else i for i in data[c]] for c,F in zip(uq_col,func)},
                **{c:data[c] for c in data.columns if c not in uq_col}
                }, index = data.index).reindex(columns = data.columns)
             for k, func in zip(it.product('lr',repeat = len(uq_col)),it.product((left,right),repeat = len(uq_col)))
            }
    models = {}
    pbar = tqdm(total = 3*2**len(uq_col),leave = (not nested), desc='Uncertain Data')
    for k,d in data_.items():
        models.update({k:LogisticRegression(**params).fit(d,results.to_numpy(dtype = bool),sample_weight)})
        pbar.update()

    n_models = models.copy()
    
    for k,m in models.items():
        B0 = m.intercept_
        B = m.coef_[0]

        nMin, nMax = _find_thresholds(B0,B,data,uq_col,catagorical = catagorical)

        for d, s in zip((nMin,nMax),('min','max')):
            d = nMin.reindex(columns = data.columns)
            n_models.update({s:LogisticRegression(**params).fit(d,results.to_numpy(dtype = bool),sample_weight)})
            pbar.update()
    pbar.close()
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

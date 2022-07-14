import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import itertools as it
from tqdm import tqdm
import pba
import scipy.optimize as so

class dslr:

    
    def __init__(self, **kwargs):

        self.models = {}
        
        self.params = LogisticRegression(**kwargs).get_params()
        
        self.__dict__.update((k, v) for k,v in self.params.items())
        self.coef_ = []
        self.intercept_ = []
        self.n_iter_ = []
        self.classes_ = None
        self.n_features_in_ = None
    
    def fit(self, data, results ,sample_weight=None):
        data_low = data.copy()
        data_hi = data.copy()
        
        for i in data.index:
            for c in data.columns:
                if data.loc[i,c].__class__.__name__ == "Interval":
                    data_low.loc[i,c] = data_low.loc[i,c].left
                    data_hi.loc[i,c] = data_hi.loc[i,c].right
                    
                    
        self.models = {
            'low': LogisticRegression(**self.params).fit(data_low,results, sample_weight),
            'hi': LogisticRegression(**self.params).fit(data_hi,results, sample_weight)
        }
        
    def predict_proba(self, X):
        
        predictions0 = np.empty((2,len(X)))
        predictions1 = np.empty((2,len(X)))
        
        
        for i, model in enumerate(self.models.values()):
            predictions0[i] = model.predict_proba(X)[:,0]
            predictions1[i] = model.predict_proba(X)[:,1]
        
        return np.array([[i.mean() for i in predictions0.T],[i.mean() for i in predictions1.T]]).T

import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
import itertools as it
from tqdm import tqdm
import pba

def split_data(features, results, test_frac = 0.5, uq_frac = 0.05, seed=random.random()):
    
    i = list(features.index)
    n = len(i)
   
    # get data indexes
    test_data_index = random.sample(i, k = int(n*test_frac))
    train_data_index = random.sample([f for f in i if f not in test_data_index], k = int((1-uq_frac) * (n-len(test_data_index))))
    uq_data_index = [f for f in i if f not in test_data_index and f not in train_data_index]

    test_data = features.loc[test_data_index]
    train_data = features.loc[train_data_index]
    uq_data = features.loc[uq_data_index]
    print('%i test data\n%i training data\n%i uncertain data' %(len(test_data_index),len(train_data_index),len(uq_data_index)))
    test_results = results.loc[test_data_index]
    train_results = results.loc[train_data_index]
  
    
    return test_data, test_results, train_data, train_results, uq_data
    
def generate_confusion_matrix(results,predictions,throw = False):

    a = 0
    b = 0
    c = 0
    d = 0
    
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
                    
    return a,b,c,d

def uc_logistic_regression(data,results,uncertain):

    models = {}

    for N,i in tqdm(enumerate(it.product([0,1],repeat=len(uncertain)))):

        new_data = pd.concat((data,uncertain), ignore_index = True)
        new_result = pd.concat((results, pd.Series(i)), ignore_index = True)

        model = LogisticRegression(max_iter=500)       
        models[str(i)] = model.fit(new_data.to_numpy(),new_result.to_numpy())
        
    return models


def ROC(model = None, predictions = None, data = None, results = None):
    
    s = []
    fpr = []
    
    if predictions is None:
        predictions = model.predict_proba(data)[:,1]
    
    for p in np.linspace(0,1,101):
        a = 0
        b = 0
        c = 0
        d = 0

        for prediction, result in zip(predictions,results):

            if prediction >= p:
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
                    
        
        s.append(a/(a+c))
        fpr.append(b/(b+d))
    return s, fpr
   
def UQ_ROC(models, data, results):
    
    s = []
    fpr = []
    
    predictions_lb = [min([m.predict_proba(data.loc[d].to_numpy().reshape(1, -1))[:,1] for k,m in models.items()]) for d in tqdm(data.index)]
    predictions_ub = [max([m.predict_proba(data.loc[d].to_numpy().reshape(1, -1))[:,1] for k,m in models.items()]) for d in tqdm(data.index)]
        
    s_lb,fpr_lb = ROC(predictions = predictions_lb, data = test_data, results = test_results)
    s_ub,fpr_ub = ROC(predictions = predictions_ub, data = test_data, results = test_results)
        
    return s_lb, fpr_lb, s_ub, fpr_ub

                  
# Import the data
heart_data = pd.read_csv('SAheart.csv',index_col = 'patient')

# Split the data into risk factors and result
factors = heart_data[['sbp','tobacco','ldl','adiposity','famhist','typea','obesity','alcohol','age']]
chd = heart_data['chd']

test_data, test_results, train_data, train_results, uq_data = split_data(factors,chd,uq_frac=0.02,seed = 2)

# Fit model
base = LogisticRegression(max_iter=500)
base.fit(train_data, train_results)

# Make preictions from test data
base_predict = base.predict(test_data)

# Make prediction for uncertain data 
uq_models = uc_logistic_regression(train_data, train_results, uq_data)

test_predict = pd.DataFrame(columns = uq_models.keys())

for key, model in uq_models.items():
    test_predict[key] = model.predict(test_data)
    
predictions = []
for i in test_predict.index:
    predictions.append([min(test_predict.loc[i]),max(test_predict.loc[i])])


## Get confusion matrix
a,b,c,d = generate_confusion_matrix(base_predict,test_results)
print('TP=%i\tFP=%i\nFN=%i\tTN=%i' %(a,b,c,d))

# Calculate sensitivity and specificity
print('Sensitivity = %.2f' %(a/(a+c)))
print('Specificity = %.2f' %(d/(b+d)))

aa,bb,cc,dd = generate_confusion_matrix(test_results,predictions)
try:
    ss = 1/(1+cc/aa)
except:
    ss = None
try:    
    tt = 1/(1+bb/dd)
except:
    tt = None
print('KEEP\na=%s\tb=%s\nc=%s\td=%s\ns=%s\tt=%s' %(aa,bb,cc,dd,ss,tt))


aaa,bbb,ccc,ddd = generate_confusion_matrix(test_results,predictions,throw = True)
try:
    sss = 1/(1+ccc/aaa)
except:
    sss = None
try:    
    ttt = 1/(1+bbb/ddd)
except:
    ttt = None
print('THROW\na=%s\tb=%s\nc=%s\td=%s\ns=%s\tt=%s' %(aaa,bbb,ccc,ddd,sss,ttt))


# ### ROC CURVE
# s,fpr = ROC(model = base, data = test_data, results = test_results)
# s_lb, fpr_lb, s_ub, fpr_ub = UQ_ROC(models = uq_models, data = test_data, results = test_results)

# plt.plot(fpr,s,'r')
# plt.plot(fpr_lb,s_lb,'g')
# plt.plot(fpr_ub,s_ub,'k')
# # print(len(fpr))

# plt.plot([0,1],[0,1],'k:')
# plt.xlabel('1-$t$')
# plt.ylabel('$s$')
# plt.show()

### PLOTS

l = len(factors.columns)
colors = ['g' if d else 'r' for c,d in train_results.iteritems()]
for i,(j,k) in enumerate(it.product(factors.columns,repeat=2)):
    if j != k:
        plt.subplot(l,l,i+1)
        plt.scatter(train_data[j],train_data[k],c=colors)
        plt.scatter(uq_data[j],uq_data[k],c='b')
        
plt.show()
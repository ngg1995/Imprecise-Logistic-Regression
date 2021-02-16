import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
import itertools as it
from tqdm import tqdm
import pba

from LRF import *

# Import the data
wine_data = pd.read_csv('winequality-red.csv',index_col = None)

# Split the data into test/train factors and result and generate uncertain points
random.seed(1111) # for reproducability

uq_data_index = random.sample([i for i in wine_data.index if wine_data.loc[i,'quality'] == 6 or wine_data.loc[i,'quality'] == 7], k = 8)
nuq_data_index = random.sample([i for i in wine_data[wine_data['quality'] <= 6].index if i not in uq_data_index], k = 150) + random.sample([i for i in wine_data[wine_data['quality'] >= 7].index if i not in uq_data_index], k = 50)
test_data_index = [i for i in wine_data.index if i not in uq_data_index and i not in nuq_data_index]

uq_data = wine_data.loc[uq_data_index,[c for c in wine_data.columns if c != 'quality']]
nuq_data = wine_data.loc[nuq_data_index,[c for c in wine_data.columns if c != 'quality']]
test_data = wine_data.loc[test_data_index,[c for c in wine_data.columns if c != 'quality']]

train_data = wine_data.loc[uq_data_index+nuq_data_index,[c for c in wine_data.columns if c != 'quality']]

nuq_results = wine_data.loc[nuq_data_index,'quality'] >= 7
train_results = wine_data.loc[train_data.index,'quality'] >= 7
test_results = wine_data.loc[test_data_index,'quality'] >= 7

### Fit logistic regression model on full dataset
base = LogisticRegression(max_iter=1000)
base.fit(train_data.to_numpy(),train_results.to_numpy())

### Fit UQ models
uq_models = uc_logistic_regression(train_data,train_results,uq_data)

### Fit models with missing data
nuq = LogisticRegression(max_iter=1000)
nuq.fit(nuq_data.to_numpy(),nuq_results.to_numpy())

### Get confusion matrix
# Classify test data
base_predict = base.predict(test_data)

# CLASSIFY NO_UQ MODEL DATA 
nuq_predict = nuq.predict(test_data)

# CLASSIFY UQ MODEL 
test_predict = pd.DataFrame(columns = uq_models.keys())

for key, model in uq_models.items():
    test_predict[key] = model.predict(test_data)
    
predictions = []
for i in test_predict.index:
    predictions.append([min(test_predict.loc[i]),max(test_predict.loc[i])])

with open('runinfo/redwine_cm.out','w') as f:
    print('TRUE MODEL',file = f)
    a,b,c,d = generate_confusion_matrix(test_results,base_predict)
    print('TP=%i\tFP=%i\nFN=%i\tTN=%i' %(a,b,c,d),file = f)

    # Calculate sensitivity and specificity
    print('Sensitivity = %.3f' %(a/(a+c)),file = f)
    print('Specificity = %.3f' %(d/(b+d)),file = f)

    print('DISCARDED DATA MODEL',file = f)
    aa,bb,cc,dd = generate_confusion_matrix(test_results,nuq_predict)
    try:
        ss = 1/(1+cc/aa)
    except:
        ss = None
    try:    
        tt = 1/(1+bb/dd)
    except:
        tt = None
    print('TP=%s\tFP=%s\nFN=%s\tTN=%s' %(aa,bb,cc,dd),file = f)

    # Calculate sensitivity and specificity
    print('Sensitivity = %.3f' %(ss),file = f)
    print('Specificity = %.3f' %(tt),file = f)
    
    print('UQ MODEL',file = f)
    aaa,bbb,ccc,ddd,eee,fff = generate_confusion_matrix(test_results,predictions,throw = True)
    try:
        sss = 1/(1+ccc/aaa)
    except:
        sss = None
    try:    
        ttt = 1/(1+bbb/ddd)
    except:
        ttt = None
        
    print('TP=%i\tFP=%i\nFN=%i\tTN=%i\nNP(+)=%i\tNP(-)=%i' %(aaa,bbb,ccc,ddd,eee,fff),file = f)

    # Calculate sensitivity and specificity
    print('Sensitivity = %.3f' %(sss),file = f)
    print('Specificity = %.3f' %(ttt),file = f)

### ROC CURVE
s,fpr = ROC(model = base, data = test_data, results = test_results)
nuq_s,nuq_fpr = ROC(model = nuq, data = test_data, results = test_results)
s_t, fpr_t, Sigma, Tau, Nu = UQ_ROC_alt(uq_models, test_data, test_results)

plt.xlabel('$1-t$')
plt.ylabel('$s$')
plt.step(fpr,s,'k', label = 'Base')
plt.step(nuq_fpr,nuq_s,'m', label = 'Discarded')
plt.step(fpr_t,s_t,'r', label = 'Not Predicting')
plt.legend()

plt.savefig('figs/redwine_ROC.png',dpi = 600)
plt.savefig('../paper/figs/redwine_ROC.png',dpi = 600)
plt.clf()
fig = plt.figure()

with open('runinfo/ex1_UC_auc.out','w') as f:
    print('NO UNCERTAINTY: %.4f' %auc(s,fpr), file = f)
    print('DISCARDED: %.4F' %auc(nuq_s,nuq_fpr),file = f)
    print('THROW: %.4f' %auc(s_t,fpr_t), file = f)


ax = plt.axes(projection='3d',elev = 45,azim = -45,proj_type = 'ortho')
ax.set_xlabel('$1-t$')
ax.set_ylabel('$s$')
ax.set_zlabel('$\\sigma,\\tau$')
ax.plot(fpr_t,s_t,'m',alpha = 0.5)
ax.plot3D(fpr,s,Sigma,'b',label = '$\\sigma$')
ax.plot3D(fpr,s,Tau,'r',label = '$\\tau$')
# ax.plot3D(fpr,s,Nu,'k',label = '$1-\\nu$')

ax.legend()

plt.savefig('figs/redwine_ROC3D.png',dpi = 600)
plt.savefig('../paper/figs/redwine_ROC3D.png',dpi = 600)

### Hosmer-Lemeshow
hl_b, pval_b = hosmer_lemeshow_test(base,train_data,train_results,Q = 20)

hl_nuq, pval_nuq = hosmer_lemeshow_test(nuq,train_data,train_results,Q = 20)

hl_min = np.inf
hl_max = 0
pval_max = 0
pval_min = 1

for k,m in uq_models.items():
    hl_,pval_ = hosmer_lemeshow_test(m,train_data,train_results,Q = 20)
    hl_min = min(hl_min,hl_)
    pval_min = min(pval_min,pval_)
    
    hl_max = max(hl_max,hl_)
    pval_max = max(pval_max,pval_)
    
    
with open('runinfo/redwine_HL.out','w') as f:
    print('base\nhl = %.3f, p = %.5f' %(hl_b,pval_b),file = f)
    print('no UQ\nhl = %.3f, p = %.5f' %(hl_nuq,pval_nuq),file = f) 

    print('UQ\nhl = [%.3f,%.3f], p = [%.5f,%.5f]' %(hl_min,hl_max,pval_min,pval_max),file = f) 
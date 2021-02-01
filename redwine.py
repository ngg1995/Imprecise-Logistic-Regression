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
uq_data_index = random.sample([i for i in wine_data.index if wine_data.loc[i,'quality'] == 6 or wine_data.loc[i,'quality'] == 7], k = 12)
train_data_index = random.sample([i for i in wine_data[wine_data['quality'] <= 6].index if i not in uq_data_index], k = 50) + random.sample([i for i in wine_data[wine_data['quality'] >= 7].index if i not in uq_data_index], k = 50)
test_data_index = [i for i in wine_data.index if i not in uq_data_index and i not in train_data_index]

uq_data = wine_data.loc[uq_data_index,[c for c in wine_data.columns if c != 'quality']]
test_data = wine_data.loc[test_data_index,[c for c in wine_data.columns if c != 'quality']]
train_data = wine_data.loc[train_data_index,[c for c in wine_data.columns if c != 'quality']]

test_results = wine_data.loc[test_data_index,'quality'] >= 7
train_results = wine_data.loc[train_data_index,'quality'] >= 7
print('prev = %.2f' %(sum(test_results)/len(wine_data)))
# # Fit model
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
with open('redwine-cm.out','w') as f:
    a,b,c,d = generate_confusion_matrix(test_results,base_predict)
    print('TP=%i\tFP=%i\nFN=%i\tTN=%i' %(a,b,c,d),file = f)

    # Calculate sensitivity and specificity
    print('Sensitivity = %.3f' %(a/(a+c)),file = f)
    print('Specificity = %.3f' %(d/(b+d)),file = f)


    aa,bb,cc,dd = generate_confusion_matrix(test_results,predictions)
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
    print('Sensitivity = %s' %(ss),file = f)
    print('Specificity = %s' %(tt),file = f)

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


## ROC CURVE

### ROC CURVE
s,fpr = ROC(model = base, data = test_data, results = test_results)
s_i, fpr_i, s_t, fpr_t = UQ_ROC(models = uq_models, data = test_data, results = test_results)

plt.plot([0,1],[0,1],'k:',label = '$s = 1-t$')
plt.plot([0,0,1],[0,1,1],'r:',label = '$s = 1-t$')
plt.xlabel('$1-t$')
plt.ylabel('$s$')
plt.step(fpr,s,'k', label = 'Base')
plt.savefig('figs/ex1_ROC.png')
plt.savefig('../paper/figs/ex1_ROC.png')


steps = 1001
X = np.linspace(0,1,steps)
Ymin = steps*[2]
Ymax = steps*[-1]

for i, x in enumerate(X):
    for k,j in zip(s_i,fpr_i):
        if j.straddles(x,endpoints = True):
            Ymin[i] = min((Ymin[i],k.Left))
            Ymax[i] = max((Ymax[i],k.Right))
            
plt.step([x for i,x in enumerate(X) if Ymax[i] != -1],[y for i,y in enumerate(Ymax) if Ymax[i] != -1],'r',label = 'Upper Bound')
plt.step([x for i,x in enumerate(X) if Ymin[i] != 2],[y for i,y in enumerate(Ymin) if Ymin[i] != 2],'b',label = 'Lower Bound')

plt.step(fpr_t,s_t,'m', label = 'Dropped Values')
plt.legend()
# # tikzplotlib.save('../paper/figs/redwine_UQ_ROC.png')
plt.savefig('figs/redwine_UQ_ROC.png')
plt.savefig('../paper/figs/redwine_UQ_ROC.png')


with open('redwine-auc.out','w') as f:
    print('NO UNCERTAINTY: %.4f' %auc(s,fpr), file = f)
    print('LOWER BOUND: %.4f' %auc([x for i,x in enumerate(X) if Ymin[i] != 2],[y for i,y in enumerate(Ymin) if Ymin[i] != 2]), file = f)
    print('UPPER BOUND: %.4f' %auc([x for i,x in enumerate(X) if Ymax[i] != 2],[y for i,y in enumerate(Ymax) if Ymax[i] != 2]), file = f)
    print('THROW: %.4f' %auc(s_t,fpr_t), file = f)
    
plt.clf()

## PLOTS

l = len(train_data.columns)
colors = ['g' if d else 'r' for c,d in train_results.iteritems()]
for i,c in enumerate(train_data.columns):
    plt.subplot(4,3,i+1)
    plt.scatter(train_data[c],train_results,marker = 'x',c=colors)
    # plt.scatter(uq_data[j],uq_data[k],c='k')
        
plt.savefig('figs/redwine.png')
plt.savefig('../paper/figs/redwine.png')


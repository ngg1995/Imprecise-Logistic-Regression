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


### ROC CURVE
s,fpr = ROC(model = base, data = test_data, results = test_results)
# s_i, fpr_i, s_t, fpr_t = UQ_ROC(models = uq_models, data = test_data, results = test_results)
s_t, fpr_t, Sigma, Tau, Nu = UQ_ROC_alt(uq_models, test_data, test_results)


# plt.plot([0,0,1],[0,1,1],'r:',label = 'Perfect Classifier')
plt.plot([0,1],[0,1],'k:',label = 'Random Classifer')
plt.xlabel('$1-t$')
plt.ylabel('$s$')
plt.step(fpr,s,'k', label = 'Base')
plt.savefig('figs/redwine_ROC.png',dpi = 600)
plt.savefig('../paper/figs/redwine_ROC.png',dpi = 600)


# steps = 1001
# X = np.linspace(0,1,steps)
# Ymin = steps*[2]
# Ymax = steps*[-1]

# for i, x in tqdm(enumerate(X)):
#     for k,j in zip(s_i,fpr_i):

#         if j.straddles(x,endpoints = True):
#             Ymin[i] = min((Ymin[i],k.Left))
#             Ymax[i] = max((Ymax[i],k.Right))

# Xmax = [0]+[x for i,x in enumerate(X) if Ymax[i] != -1]+[1]
# Xmin = [0]+[x for i,x in enumerate(X) if Ymin[i] != 2]+[1]
# Ymax = [0]+[y for i,y in enumerate(Ymax) if Ymax[i] != -1]+[1]
# Ymin = [0]+[y for i,y in enumerate(Ymin) if Ymin[i] != 2]+[1]

# plt.step(Xmax,Ymax,'r',label = 'Upper Bound',where = 'pre')
# plt.step(Xmin,Ymin,'b',label = 'Lower Bound',where = 'post')


plt.step(fpr_t,s_t,'m', label = 'Not Predicting')
plt.legend()

# tikzplotlib.save('../paper/figs/redwine_UQ_ROC.png')
plt.savefig('figs/redwine_UC_ROC.png',dpi = 600)
plt.savefig('../paper/figs/redwine_UC_ROC.png',dpi = 600)

# plt.clf()

with open('redwine-auc.out','w') as f:
    print('NO UNCERTAINTY: %.4f' %auc(s,fpr), file = f)
    # print('NO UNCERTAINTY: %.4f' %roc_auc_score(base.predict_proba(test_data)[:,1],test_results), file = f)
    # print('LOWER BOUND: %.4f' %auc(Ymin,Xmin), file = f)
    # print('UPPER BOUND: %.4f' %auc(Ymax,Xmax), file = f)
    print('THROW: %.4f' %auc(s_t,fpr_t), file = f)


######
from mpl_toolkits import mplot3d

fig = plt.figure()

ax = plt.axes(projection='3d',elev = 45,azim = -45,proj_type = 'ortho')
ax.set_xlabel('$1-t$')
ax.set_ylabel('$s$')
# ax.set_zlabel('$1-\sigma,1-\\tau$')



ax.plot3D(fpr_t,s_t,Sigma,'r',label = '$\\sigma$')
ax.plot3D(fpr_t,s_t,Tau,'g',label = '$\\tau$')
# ax.plot3D(fpr,s,Nu,'k',label = '$1-\\nu$')
ax.plot(fpr_t,s_t,'k',alpha = 0.5)


ax.legend()

plt.savefig('figs/redwine_ROC3D.png')
plt.savefig('../paper/figs/redwine_ROC3D.png')

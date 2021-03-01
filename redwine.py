import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
import itertools as it
from tqdm import tqdm
import pba

from LRF import *

### Import the data
wine_data = pd.read_csv('winequality-red.csv',index_col = None,usecols = ['volatile acidity','citric acid','chlorides','pH','sulphates','alcohol','quality'])

# Split the data into test/train factors and result and generate uncertain points
random.seed(1111) # for reproducability

uq_data_index = random.sample([i for i in wine_data.index if wine_data.loc[i,'quality'] == 6 or wine_data.loc[i,'quality'] == 7], k = 10)
nuq_data_index = random.sample([i for i in wine_data[wine_data['quality'] <= 6].index if i not in uq_data_index], k = 100) + random.sample([i for i in wine_data[wine_data['quality'] >= 7].index if i not in uq_data_index], k = 100)
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
print(*zip(train_data.columns,*base.coef_))
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
    
    aaai,bbbi,ccci,dddi = generate_confusion_matrix(test_results,predictions,throw = False)
    try:
        sssi = 1/(1+ccci/aaai)
    except:
        sssi = None
    try:    
        ttti = 1/(1+bbbi/dddi)
    except:
        ttti = None
        
    print('TP=[%i,%i]\tFP=[%i,%i]\nFN=[%i,%i]\tTN=[%i,%i]' %(*aaai,*bbbi,*ccci,*dddi),file = f)

    # Calculate sensitivity and specificity
    print('Sensitivity = [%.3f,%.3f]\nSpecificity = [%.3f,%.3f]' %(*sssi,*ttti),file = f)
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

### Descriminatory Performance Plots
s,fpr,predictions = ROC(model = base, data = train_data, results = train_results)
nuq_s,nuq_fpr,nuq_predictions = ROC(model = nuq, data = train_data, results = train_results)
s_t, fpr_t, Sigma, Tau, Nu = UQ_ROC_alt(uq_models, train_data, train_results)

s_i, fpr_i,uq_predictions = UQ_ROC(uq_models, train_data, train_results)

densfig,axdens = plt.subplots(nrows = 2, sharex= True)

for i,(p,u,nuqp,r) in enumerate(zip(predictions,uq_predictions,nuq_predictions,train_results.to_list())):
    yd = np.random.uniform(-0.11,-.31)
    if r:
        axdens[0].scatter(p,np.random.uniform(-0.1,0.1),color = 'k',marker = 'o',alpha = 0.5)
        axdens[0].scatter(nuqp,np.random.uniform(0.11,0.31),color = '#DC143C',marker = 'o',alpha = 0.5)
        axdens[0].plot([u[0],u[1]],[yd,yd],color = '#4169E1',alpha = 0.3)
        axdens[0].scatter([u[0],u[1]],[yd,yd],color = '#4169E1',marker = '|')
    else:
        axdens[1].scatter(p,np.random.uniform(-.1,0.1),color = 'k',marker = 'o',alpha = 0.5)
        axdens[1].scatter(nuqp,np.random.uniform(0.11,.31),color = '#DC143C',marker = 'o',alpha = 0.5)
        axdens[1].plot([u[0],u[1]],[yd,yd],color = '#4169E1',alpha = 0.3)
        axdens[1].scatter([u[0],u[1]],[yd,yd],color = '#4169E1',marker = '|')
        
        
axdens[0].set(ylabel = 'Outcome = 1',yticks = [])
axdens[1].set(xlabel = '$\pi$',ylabel = 'Outcome = 0',yticks = [])
densfig.tight_layout()

rocfig,axroc = plt.subplots(1,1)
axroc.plot([0,1],[0,1],'k:',label = 'Random Classifier')
axroc.set(xlabel = '$fpr$',ylabel='$s$')
axroc.plot(fpr,s,'k',label = 'Base')
axroc.plot(nuq_fpr,nuq_s,color='#DC143C',linestyle='--',label='No Uncertainty')
axroc.plot(fpr_t,s_t,'#4169E1',label='Uncertain (No prediction)')
axroc.legend()
rocfig.savefig('figs/redwine_ROC.png',dpi = 600)
rocfig.savefig('../paper/figs/redwine_ROC.png',dpi = 600)
densfig.savefig('figs/redwine_dens.png',dpi =600)
densfig.savefig('../paper/figs/redwine_dens.png',dpi =600)


with open('runinfo/redwine_auc.out','w') as f:
    print('NO UNCERTAINTY: %.3f' %auc(s,fpr), file = f)
    print('MIDPOINTS: %.4F' %auc(nuq_s,nuq_fpr),file = f)
    print('THROW: %.3f' %auc(s_t,fpr_t), file = f)
    # print('INTERVALS: [%.3f,%.3f]' %(auc_int_min,auc_int_max), file = f)
    


fig = plt.figure()
ax = plt.axes(projection='3d',elev = 45,azim = -45,proj_type = 'ortho')
ax.set_xlabel('$fpr$')
ax.set_ylabel('$s$')
# ax.set_zlabel('$1-\sigma,1-\\tau$')
ax.plot(fpr_t,s_t,'#4169E1',alpha = 0.5)
ax.plot3D(fpr,s,Sigma,'#FF8C00',label = '$\\sigma$')
ax.plot3D(fpr,s,Tau,'#008000',label = '$\\tau$')
# ax.plot3D(fpr,s,Nu,'k',label = '$1-\\nu$')

ax.legend()

plt.savefig('figs/redwine_ROC3D.png',dpi = 600)
plt.savefig('../paper/figs/redwine_ROC3D.png',dpi = 600)
plt.clf()

plt.xlabel('$fpr$/$s$')
plt.ylabel('$\\sigma$/$\\tau$')
plt.plot(s,Sigma,'#FF8C00',label = '$\\sigma$ v $s$')
plt.plot(fpr,Tau,'#008000',label = '$\\tau$ v $fpr$')
plt.legend()

plt.savefig('figs/redwine_ST.png',dpi = 600)
plt.savefig('../paper/figs/redwine_ST.png',dpi = 600)

plt.clf()

### Hosmer-Lemeshow
hl_b, pval_b = hosmer_lemeshow_test(base,train_data,train_results,g = 10)

hl_nuq, pval_nuq = hosmer_lemeshow_test(nuq,train_data,train_results,g = 10)

hl_uq, pval_uq = UQ_hosmer_lemeshow_test(uq_models,train_data,train_results,g = 10)

with open('runinfo/redwine_HL.out','w') as f:
    print('base\nhl = %.3f, p = %.3f' %(hl_b,pval_b),file = f)
    print('no UQ\nhl = %.3f, p = %.3f' %(hl_nuq,pval_nuq),file = f) 
    print('UQ\nhl = [%.3f,%.3f], p = [%.3f,%.3f]' %(*hl_uq,*pval_uq),file = f)
    
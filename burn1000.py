import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
import itertools as it
from tqdm import tqdm
import pba

from LRF import *

def intervalise(val,eps,method='u',b=0,bounds = None):
    
    if method == 'u':
        m = np.random.uniform(val-eps,val+eps)
    elif method == 't':
        m = np.random.triangular(val-eps,val-b*eps,val+eps)
    
    if bounds is not None:
        if m-eps < bounds[0]:
            return pba.I(bounds[0],m+eps)
        elif m+eps >bounds[1]:
            return pba.I(m-eps,bounds[1])
        
    return pba.I(m-eps,m+eps)


def midpoints(data, binary_cols):
    n_data = data.copy()
    for c in data.columns:
        if c in binary_cols:
            continue
        for i in data.index:
            if data.loc[i,c].__class__.__name__ == 'Interval':
                n_data.loc[i,c] = data.loc[i,c].midpoint()
            
    return n_data

### Import data
data = pd.read_table('burn1000.txt',index_col = 'ID')

results = data['DEATH']
train_data = data[[c for c in data.columns if c not in ['DEATH','FACILITY']]]
UQdata = train_data.copy()

# Split the data into test/train factors and result and generate uncertain points
random.seed(25) # for reproducability

## Select some data to be intervalised
binary_cols = ['RACE','FLAME']
binary_uq ={c: random.sample([i for i in data.index], k = n) for c,n in zip(binary_cols,(50,50))}
binary_index = {i for i in it.chain(*binary_uq.values())}
eps = {
    #    "AGE":(1,'u',0,(0,np.inf)),
       "TBSA":(2,'t',-1,(0,100))
       }
UQdata = pd.DataFrame({
    **{c: [pba.I(0,1) if i in binary_uq[c] else train_data.loc[i,c] for i in data.index] for c in binary_cols},
    **{c:[intervalise(train_data.loc[i,c],*eps[c]) for i in train_data.index] for c, e in eps.items()},
    **{c: train_data[c] for c in train_data.columns if c not in (*binary_cols,*eps.keys())}
    }, index = data.index, dtype = 'O').reindex(columns = train_data.columns)

print(len(binary_index))
### Fit logistic regression model on full dataset
base = LogisticRegression(max_iter=1000)
base.fit(train_data.to_numpy(),results.to_numpy())
print(train_data.columns,base.coef_)

### Fit models with none UQ data data
nuq_data =  midpoints(UQdata.loc[[i for i in data.index if i not in binary_index]], binary_cols)
nuq_results = results.loc[[i for i in data.index if i not in binary_index]]
nuq = LogisticRegression(max_iter=1000)
nuq.fit(nuq_data.to_numpy(),nuq_results.to_numpy())
print(nuq_data.columns,nuq.coef_)

### Fit UQ models
uq_models = int_logistic_regression(UQdata,results,binary_cols = binary_cols)

### Get confusion matrix
# Classify test data
base_predict = base.predict(train_data)

# CLASSIFY NO_UQ MODEL DATA 
nuq_predict = nuq.predict(train_data)

# CLASSIFY UQ MODEL 
train_predict = pd.DataFrame(columns = uq_models.keys())

for key, model in uq_models.items():
    train_predict[key] = model.predict(train_data)
    
predictions = []
for i in train_predict.index:
    predictions.append([min(train_predict.loc[i]),max(train_predict.loc[i])])

with open('runinfo/burn1000_cm.out','w') as f:
    print('TRUE MODEL',file = f)
    a,b,c,d = generate_confusion_matrix(results,base_predict)
    print('TP=%i\tFP=%i\nFN=%i\tTN=%i' %(a,b,c,d),file = f)

    # Calculate sensitivity and specificity
    print('Sensitivity = %.3f' %(a/(a+c)),file = f)
    print('Specificity = %.3f' %(d/(b+d)),file = f)

    print('DISCARDED DATA MODEL',file = f)
    aa,bb,cc,dd = generate_confusion_matrix(results,nuq_predict)
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
    
    aaai,bbbi,ccci,dddi = generate_confusion_matrix(results,predictions,throw = False)
    print(aaai,bbbi,ccci,dddi)
    try:
        sssi = 1/(1+ccci/aaai)
    except:
        sssi = None
    try:    
        ttti = 1/(1+bbbi/dddi)
    except:
        ttti = None
        
    print('TP=%s\tFP=%s\nFN=%s\tTN=%s' %(aaai,bbbi,ccci,dddi),file = f)

    # Calculate sensitivity and specificity
    print('Sensitivity = [%.3f,%.3f]\nSpecificity = [%.3f,%.3f]' %(*sssi,*ttti),file = f)

    aaa,bbb,ccc,ddd,eee,fff = generate_confusion_matrix(results,predictions,throw = True)
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
s,fpr,predictions = ROC(model = base, data = train_data, results = results)


nuq_s,nuq_fpr,nuq_predictions = ROC(model = nuq, data = train_data, results = results)
s_t, fpr_t, Sigma, Tau, Nu = UQ_ROC_alt(uq_models, train_data, results)

s_i, fpr_i,uq_predictions = UQ_ROC(uq_models, train_data, results)


densfig,axdens = plt.subplots(nrows = 2, sharex= True)

for i,(p,u,nuqp,r) in enumerate(zip(predictions,uq_predictions,nuq_predictions,results.to_list())):
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
rocfig.savefig('figs/burn1000_ROC.png',dpi = 600)
rocfig.savefig('../paper/figs/burn1000_ROC.png',dpi = 600)
densfig.savefig('figs/burn1000_dens.png',dpi =600)
densfig.savefig('../paper/figs/burn1000_dens.png',dpi =600)


with open('runinfo/burn1000_auc.out','w') as f:
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
ax.plot3D(fpr_t,s_t,Sigma,'#FF8C00',label = '$\\sigma$')
ax.plot3D(fpr_t,s_t,Tau,'#008000',label = '$\\tau$')
# ax.plot3D(fpr,s,Nu,'k',label = '$1-\\nu$')

ax.legend()

plt.savefig('figs/burn1000_ROC3D.png',dpi = 600)
plt.savefig('../paper/figs/burn1000_ROC3D.png',dpi = 600)
plt.clf()

plt.xlabel('$fpr$/$s$')
plt.ylabel('$\\sigma$/$\\tau$')
plt.plot(s_t,Sigma,'#FF8C00',label = '$\\sigma$ v $s$')
plt.plot(fpr_t,Tau,'#008000',label = '$\\tau$ v $fpr$')
plt.legend()

plt.savefig('figs/burn1000_ST.png',dpi = 600)
plt.savefig('../paper/figs/burn1000_ST.png',dpi = 600)

plt.clf()


### Hosmer-Lemeshow
hl_b, pval_b = hosmer_lemeshow_test(base,train_data,results,g = 10)

hl_nuq, pval_nuq = hosmer_lemeshow_test(nuq,train_data,results,g = 10)

hl_uq, pval_uq = UQ_hosmer_lemeshow_test(uq_models,train_data,results,g = 10)

    
    
with open('runinfo/burn1000_HL.out','w') as f:
    print('base\nhl = %.3f, p = %.5f' %(hl_b,pval_b),file = f)
    print('no UQ\nhl = %.3f, p = %.5f' %(hl_nuq,pval_nuq),file = f) 
    print('UQ\nhl = [%.3f,%.3f], p = [%.3f,%.3f]' %(*hl_uq,*pval_uq),file = f) 
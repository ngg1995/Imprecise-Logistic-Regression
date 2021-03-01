import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
import itertools as it
from tqdm import tqdm
import pba
import tikzplotlib
import random
def histogram(probs, dif = 0, uq = False, bars = 10):
    x = np.arange(bars)/bars
    if uq: 
        low_height = bars*[0]
        hi_height = bars*[0]
    else:
        height = bars*[0]
    for p in probs:
        for i,j in reversed(list(enumerate(x))):
            if uq:
                if i + 1 == bars:
                    if p[0] > j:
                        low_height[i] += 1
                        hi_height[i] += 1
                        break
                    if p[1] > j:
                        hi_height[i] += 1
                else:
                    if p[0] > j and p[1] < x[i+1]:
                        low_height[i] += 1
                        hi_height[i] += 1
                        break
                    if p[1] > j:
                        hi_height[i] += 1
                    if p[0] > j:
                        hi_height[i] += 1
                        break
            else:
                if p > j:
                    height[i] += 1
                    break
            
    if dif != 0:
        x = [i+dif for i in x]    
    
    if uq:
        return x,low_height, hi_height
    return x, height
from LRF import *

def intervalise(val,eps,method,b=0.5,bounds = None):
    
    if method == 'u':
        m = np.random.uniform(val-eps,val+eps)
    elif method == 'b':
        m = val - eps + 2*b*eps
    elif method == 't':

        m = np.random.triangular(val-eps,val-b*eps,val+eps)
    
    if bounds is not None:
        if m-eps < bounds[0]:
            return pba.I(bounds[0],m+eps)
        elif m+eps >bounds[1]:
            return pba.I(m-eps,bounds[1])
        
    return pba.I(m-eps,m+eps)


def midpoints(data):
    n_data = data.copy()
    for c in data.columns:
        for i in data.index:
            if data.loc[i,c].__class__.__name__ == 'Interval':

                n_data.loc[i,c] = data.loc[i,c].midpoint()

            
    return n_data

def generate_results(data):
    # set seed for reproducability
    np.random.seed(10)
    results = pd.Series(index = data.index, dtype = 'bool')
    
    for i in data.index:

        results.loc[i] = data.loc[i,0] >= 5+2*np.random.randn()    
        
    return results


### Generate Data
# set seed for reproducability
np.random.seed(1)
random.seed(2)

# Params
some = 50 # training datapoints
many = 500 # many test samples

train_data = pd.DataFrame(10*np.random.rand(some,1))
train_results = generate_results(train_data)

test_data = pd.DataFrame(10*np.random.rand(many,1))
test_results = generate_results(test_data)

### Fit logistic regression model
base = LogisticRegression()
base.fit(train_data.to_numpy(),train_results.to_numpy())

# Intervalise data
eps = 0.25

UQdata = pd.DataFrame({
    0:[intervalise(train_data.iloc[i,0],eps,'t',0.75,(0,10)) for i in train_data.index]
    }, dtype = 'O')

### Fit logistic regression model on full dataset
base = LogisticRegression(max_iter=1000)
base.fit(train_data.to_numpy(),train_results.to_numpy())


### Fit models with midpoint data
nuq_data = midpoints(UQdata)
nuq = LogisticRegression(max_iter=1000)
nuq.fit(nuq_data.to_numpy(),train_results.to_numpy())

### Fit UQ models
uq_models = int_logistic_regression(UQdata,train_results)


### Plot results
steps = 300
lX = np.linspace(0,10,steps)
lY = base.predict_proba(lX.reshape(-1, 1))[:,1]
lYn = nuq.predict_proba(lX.reshape(-1, 1))[:,1]

plt.xlabel('$x$')
plt.ylabel('$\pi_x$')
plt.plot(lX,lY,color='k',zorder=10,lw=2)
plt.plot(lX,lYn,color='m',zorder=10,lw=2)

for u,m,r in zip(UQdata[0],train_data[0],train_results.to_list()):
    yd = np.random.uniform(-0.05,0.05)
    # plt.plot(m,r+yd,color = 'b',marker = 'x')
    plt.plot([u.Left,u.Right],[r+yd,r+yd],color = 'b', marker='|')

lYmin = np.ones(steps)
lYmax = np.zeros(steps)

for n, model in uq_models.items():
    lY = model.predict_proba(np.linspace(0,10,steps).reshape(-1, 1))[:,1]
    lYmin = [min(i,j) for i,j in zip(lY,lYmin)]
    lYmax = [max(i,j) for i,j in zip(lY,lYmax)]
    plt.plot(lX,lY,color = 'grey',alpha = 0.2,lw = 0.5)

plt.plot(lX,lYmax,color='red',lw=2)
plt.plot(lX,lYmin,color='red',lw=2)

plt.savefig('../paper/figs/ex1_int.png',dpi = 600)
plt.savefig('figs/ex1_int.png',dpi = 600)

plt.clf()


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

with open('runinfo/ex1_int_cm.out','w') as f:
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


### ROC CURVE
s,fpr,predictions = ROC(model = base, data = test_data, results = test_results)
nuq_s,nuq_fpr,nuq_predictions = ROC(model = nuq, data = test_data, results = test_results)
s_t, fpr_t, Sigma, Tau, Nu = UQ_ROC_alt(uq_models, test_data, test_results)

s_i, fpr_i,uq_predictions = UQ_ROC(uq_models, test_data, test_results)

rocfig,ax = plt.subplots(2,2)

ax[0,0].scatter(predictions,test_results+np.random.uniform(-0.05,0.05,len(predictions)),marker = '.',color='k',alpha = 0.2)
ax[0,0].scatter(nuq_predictions,test_results+np.random.uniform(0.1,0.2,len(predictions)),marker = '.',color='m',alpha = 0.2)
ax[0,0].set(xlabel = '$\pi$',ylabel = 'Outcome',yticks = [0,1])
for u,r in zip(uq_predictions,test_results.to_list()):
    yd = np.random.uniform(-0.1,-0.2)
    # plt.plot(m,r+yd,color = 'b',marker = 'x')
    ax[0,0].plot([u[0],u[1]],[r+yd,r+yd],color = 'b',alpha = 0.1)

ax[1,0].plot([0,1],[0,1],'k:',label = 'Random Classifier')
ax[1,0].set(xlabel = '$fpr$',ylabel='$s$')
ax[1,0].plot(fpr,s,'k')
ax[1,0].plot(nuq_fpr,nuq_s,'m--')
ax[1,0].plot(fpr_t,s_t,'b')


ax[0,1].bar(*histogram([p for p,r in zip(predictions,test_results) if r]),width = 0.0333,color = 'k',align = 'edge')
ax[0,1].bar(*histogram([p for p,r in zip(nuq_predictions,test_results) if r],dif = 0.0333),width = 0.0333,color = 'm',align = 'edge')
x,low_height, hi_height = histogram([p for p,r in zip(uq_predictions,test_results) if r],uq=True,dif = 0.0666)
ax[0,1].bar(x,hi_height,width = 0.0333,color = 'w',edgecolor = 'b',align = 'edge')
ax[0,1].bar(x,low_height,width = 0.0333,color = 'b',align = 'edge')

ax[0,1].set(title = 'Outcome = 1',xlabel = '$\pi$',ylabel = 'Density',xticks = np.linspace(0,1,11),xticklabels = [0,'',.2,'',.4,'',.6,'',.8,'',1])
ax[0,1].yaxis.set_label_position("right")
ax[0,1].yaxis.tick_right()

ax[1,1].bar(*histogram([p for p,r in zip(predictions,test_results) if not r]),width = 0.0333,color = 'k',align = 'edge')
ax[1,1].bar(*histogram([p for p,r in zip(nuq_predictions,test_results) if not r],dif = 0.0333),width = 0.0333,color = 'm',align = 'edge')
x,low_height, hi_height = histogram([p for p,r in zip(uq_predictions,test_results) if not r],uq=True,dif = 0.0666)
ax[1,1].bar(x,hi_height,width = 0.0333,color = 'w',edgecolor = 'b',align = 'edge')
ax[1,1].bar(x,low_height,width = 0.0333,color = 'b',align = 'edge')

ax[1,1].set(title = 'Outcome = 0',xlabel = '$\pi$',ylabel = 'Density',xticks = np.linspace(0,1,11),xticklabels = [0,'',.2,'',.4,'',.6,'',.8,'',1])
ax[1,1].yaxis.set_label_position("right")
ax[1,1].yaxis.tick_right()

rocfig.tight_layout()
rocfig.savefig('figs/ex1_int_ROC.png',dpi = 600)
rocfig.savefig('../paper/figs/ex1_int_ROC.png',dpi = 600)

with open('runinfo/ex1_int_auc.out','w') as f:
    print('NO UNCERTAINTY: %.3f' %auc(s,fpr), file = f)
    print('MIDPOINTS: %.4F' %auc(nuq_s,nuq_fpr),file = f)
    print('THROW: %.3f' %auc(s_t,fpr_t), file = f)
    # print('INTERVALS: [%.3f,%.3f]' %(auc_int_min,auc_int_max), file = f)
    


fig = plt.figure()
ax = plt.axes(projection='3d',elev = 45,azim = -45,proj_type = 'ortho')
ax.set_xlabel('$fpr$')
ax.set_ylabel('$s$')
# ax.set_zlabel('$1-\sigma,1-\\tau$')
ax.plot(fpr_t,s_t,'m',alpha = 0.5)
ax.plot3D(fpr,s,Sigma,'b',label = '$\\sigma$')
ax.plot3D(fpr,s,Tau,'r',label = '$\\tau$')
# ax.plot3D(fpr,s,Nu,'k',label = '$1-\\nu$')

ax.legend()

plt.savefig('figs/ex1_int_ROC3D.png',dpi = 600)
plt.savefig('../paper/figs/ex1_int_ROC3D.png',dpi = 600)
plt.clf()

plt.xlabel('$fpr$/$s$')
plt.ylabel('$\\sigma$/$\\tau$')
plt.plot(s,Sigma,'g',label = '$\\sigma$ v $s$')
plt.plot(fpr,Tau,'r',label = '$\\tau$ v $t$')
plt.legend()

plt.savefig('figs/ex1_int_ST.png',dpi = 600)
plt.savefig('../paper/figs/ex1_int_ST.png',dpi = 600)

plt.clf()

### Hosmer-Lemeshow
hl_b, pval_b = hosmer_lemeshow_test(base,train_data,train_results,g = 10)

hl_nuq, pval_nuq = hosmer_lemeshow_test(nuq,train_data,train_results,g = 10)

hl_uq, pval_uq = UQ_hosmer_lemeshow_test(uq_models,train_data,train_results,g = 10)


with open('runinfo/ex1_int_HL.out','w') as f:
    print('base\nhl = %.3f, p = %.3f' %(hl_b,pval_b),file = f)
    print('Midpoints\nhl = %.3f, p = %.3f' %(hl_nuq,pval_nuq),file = f) 
    print('UQ\nhl = %s, p = %s' %(hl_uq,pval_uq),file = f) 
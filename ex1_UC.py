import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import itertools as it
from tqdm import tqdm
import pba
import tikzplotlib
from LRF import *

def generate_results(data):

    results = pd.Series(index = data.index, dtype = 'bool')
    for row in data.index:

        results[row] = sum(data.loc[row]) >= len(data.columns)*(15+5*np.random.randn())
    
    return results


# set seed to ensure same data
np.random.seed(10)

# Params
many = 50
dim = 1
few = 5
some = 1000

# Generate data
data = pd.DataFrame(40*np.random.rand(many,dim))
results = generate_results(data)

# Generate uncertain points
uncertain = 15+pd.DataFrame(5*np.random.rand(few,dim))

# Generate test data 
np.random.seed(111)
test_data = pd.DataFrame(40*np.random.rand(some,dim))
test_results = generate_results(test_data)

# Fit base model
base = LogisticRegression()
base.fit(data.to_numpy(),results.to_numpy())

# Classify test data
base_predict = base.predict(test_data)

# Fit model
uq_models = uc_logistic_regression(data,results,uncertain)

# Classify test data
test_predict = pd.DataFrame(columns = uq_models.keys())

for key, model in uq_models.items():
    test_predict[key] = model.predict(test_data)
    
predictions = []
for i in test_predict.index:
    predictions.append([min(test_predict.loc[i]),max(test_predict.loc[i])])

# Fit base model
base = LogisticRegression()
base.fit(data.to_numpy(),results.to_numpy())

# Classify test data
base_predict = base.predict(test_data)
print(base.coef_)

# # Plot results
steps = 300
lX = np.linspace(data.min(),data.max(),steps)
lY = base.predict_proba(lX.reshape(-1, 1))[:,1]

plt.xlabel('X')
plt.ylabel('$\Pr(Y=1|X)$')
plt.scatter(data,results,color='blue',zorder=10)
# plt.scatter(test_data,test_results,color='green',zorder=10)
plt.plot(lX,lY,color='k',zorder=10,lw=2)
plt.savefig('../paper/figs/ex1.png',dpi = 600)
plt.savefig('figs/ex1.png',dpi = 600)


for x in uncertain[0]:

    plt.plot([x,x],[0,1],color='blue')

lYmin = np.ones(steps)
lYmax = np.zeros(steps)

for n, model in uq_models.items():
    lY = model.predict_proba(np.linspace(data.min(),data.max(),steps).reshape(-1, 1))[:,1]
    lYmin = [min(i,j) for i,j in zip(lY,lYmin)]
    lYmax = [max(i,j) for i,j in zip(lY,lYmax)]
    plt.plot(lX,lY,color = 'grey')


plt.plot(lX,lYmax,color='red',lw=2)
plt.plot(lX,lYmin,color='red',lw=2)

plt.savefig('../paper/figs/ex1_UC.png',dpi = 600)
plt.savefig('figs/ex1_UC.png',dpi = 600)

plt.clf()

## Get confusion matrix
with open('runinfo/ex1_UC_cm.out','w') as f:
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
s_t, fpr_t, Sigma, Tau, Nu = UQ_ROC_alt(uq_models, test_data, test_results)

plt.plot([0,1],[0,1],'k:',label = 'Random Classifier')
plt.plot([0],[1],'ro',label = 'Perfect Classifier')
plt.xlabel('$1-t$')
plt.ylabel('$s$')
plt.plot(fpr,s,'k', label = 'Base')
plt.legend()
plt.savefig('figs/ex1_ROC.png',dpi = 600)
plt.savefig('../paper/figs/ex1_ROC.png',dpi = 600)


steps = 1001
X = np.linspace(0,1,steps)
Ymin = steps*[2]
Ymax = steps*[-1]

plt.step(fpr_t,s_t,'m', label = 'Not Predicting')
plt.legend()

# tikzplotlib.save('../paper/figs/ex1_UQ_ROC.png')
plt.savefig('figs/ex1_UC_ROC.png',dpi = 600)
plt.savefig('../paper/figs/ex1_UC_ROC.png',dpi = 600)

# plt.clf()

with open('runinfo/ex1_UC_auc.out','w') as f:
    print('NO UNCERTAINTY: %.4f' %auc(s,fpr), file = f)
    # print('NO UNCERTAINTY: %.4f' %roc_auc_score(base.predict_proba(test_data)[:,1],test_results), file = f)
    # print('LOWER BOUND: %.4f' %auc(Ymin,Xmin), file = f)
    # print('UPPER BOUND: %.4f' %auc(Ymax,Xmax), file = f)
    print('THROW: %.4f' %auc(s_t,fpr_t), file = f)



######

fig = plt.figure()

ax = plt.axes(projection='3d',elev = 45,azim = -45,proj_type = 'ortho')
ax.set_xlabel('$1-t$')
ax.set_ylabel('$s$')
# ax.set_zlabel('$1-\sigma,1-\\tau$')
ax.plot(fpr_t,s_t,'m',alpha = 0.5)
ax.plot3D(fpr,s,Sigma,'b',label = '$\\sigma$')
ax.plot3D(fpr,s,Tau,'r',label = '$\\tau$')
# ax.plot3D(fpr,s,Nu,'k',label = '$1-\\nu$')

ax.legend()

plt.savefig('figs/ex1_UC_ROC3D.png',dpi = 600)
plt.savefig('../paper/figs/ex1_UC_ROC3D.png',dpi = 600)
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
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

def intervalise(val,eps):
    r = np.random.rand()
    return pba.I(val - r*eps, val + (1-r)*eps)
    
# Load data
heart_data = pd.read_csv('SAheart.csv',index_col='patient')

data = heart_data[[c for c in heart_data.columns if c != 'chd']]
results = heart_data['chd']

train_data, test_data, train_results, test_results = train_test_split(data,results, test_size = 0.5,random_state = 0)

# Intervalise data
eps = {'adiposity':0.5}
np.random.seed(0)
UQdata = pd.DataFrame({
    **{'adiposity':[intervalise(train_data.loc[i,'adiposity'],eps['adiposity']) for i in train_data.index]},
    **{c:train_data[c] for c in train_data.columns if c != 'adiposity'}
    }, dtype = 'O')

# Fit base model
base = LogisticRegression(max_iter = 1000)
base.fit(train_data.to_numpy(),train_results.to_numpy())

# Classify test data
base_predict = base.predict(test_data)

## fit interval model
uq_models = int_logistic_regression(UQdata,train_results)

## Test estimated vs Monte Carlo
# ir, oor = check_int_MC(uq_models,UQdata,train_results,1000,test_data)
# with open('runinfo/heart-MCtest.out','w') as f:
#     print('in bounds %i,%.3f\nout %i,%.3f'%(ir,(ir/(ir+oor)),oor,(oor/(ir+oor))),file = f)

# Classify test data
test_predict = pd.DataFrame(columns = uq_models.keys())

for key, model in uq_models.items():
    test_predict[key] = model.predict(test_data)
    
predictions = []
for i in test_predict.index:
    predictions.append([min(test_predict.loc[i]),max(test_predict.loc[i])])


## Get confusion matrix
with open('runinfo/heart-cm.out','w') as f:
    a,b,c,d = generate_confusion_matrix(test_results,base_predict)
    print('TP=%i\tFP=%i\nFN=%i\tTN=%i' %(a,b,c,d),file = f)

    # Calculate sensitivity and specificity
    print('Sensitivity = %.3f' %(a/(a+c)),file = f)
    print('Specificity = %.3f' %(d/(b+d)),file = f)

    aaa,bbb,ccc,ddd,eee,fff = generate_confusion_matrix(test_results,predictions,throw = True)
    try:
        sss = 1/(1+ccc/aaa)
    except:
        sss = None
    try:    
        ttt = 1/(1+bbb/ddd)
    except:
        ttt = 0
        print(ddd,bbb)
        
    print('TP=%i\tFP=%i\nFN=%i\tTN=%i\nNP(+)=%i\tNP(-)=%i' %(aaa,bbb,ccc,ddd,eee,fff),file = f)
    
    # Calculate sensitivity and specificity
    print('Sensitivity = %.3f' %(sss),file = f)
    print('Specificity = %.3f' %(ttt),file = f)
    print('sigma = %3f' %(eee/(aaa+ccc+eee)),file = f)
    print('tau = %3f' %(fff/(bbb+ddd+fff)),file = f)

### ROC CURVE
s,fpr = ROC(model = base, data = test_data, results = test_results)
s_t, fpr_t, Sigma, Tau, Nu = UQ_ROC_alt(uq_models, test_data, test_results)
plt.step([0,1],[0,1],'k:',label = 'Random Classifer')
plt.xlabel('$1-t$')
plt.ylabel('$s$')
plt.step(fpr,s,'k', label = 'Base')


steps = 1001
X = np.linspace(0,1,steps)
Ymin = steps*[2]
Ymax = steps*[-1]

plt.step(fpr_t,s_t,'m', label = 'Not Predicting')
plt.legend()

plt.savefig('figs/heart_ROC.png',dpi = 600)
plt.savefig('../paper/figs/heart_ROC.png',dpi = 600)

plt.clf()

with open('runinfo/heart-auc.out','w') as f:
    print('NO UNCERTAINTY: %.4f' %auc(s,fpr), file = f)
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

plt.savefig('figs/heart_ROC3D.png',dpi = 600)
plt.savefig('../paper/figs/heart_ROC3D.png',dpi = 600)
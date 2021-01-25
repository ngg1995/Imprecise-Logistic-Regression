import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics as skm
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
# np.random.seed(10)

# Params
many = 25
dim = 1
few = 5
some = 100

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
models = uc_logistic_regression(data,results,uncertain)

# Classify test data
test_predict = pd.DataFrame(columns = models.keys())

for key, model in models.items():
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
lX = np.linspace(data.min(),data.max(),100)
lY = base.predict_proba(lX.reshape(-1, 1))[:,1]

plt.xlabel('X')
plt.ylabel('$\Pr(Y=1|X)$')
plt.scatter(data,results,color='blue',zorder=10)
plt.plot(lX,lY,color='k',zorder=10,lw=2)
plt.savefig('../paper/figs/ex1.png')
plt.savefig('figs/ex1.png')


for x in uncertain[0]:

    plt.plot([x,x],[0,1],color='blue')

lYmin = np.ones(300)
lYmax = np.zeros(300)

for n, model in models.items():
    lY = model.predict_proba(np.linspace(data.min(),data.max(),100).reshape(-1, 1))[:,1]
    lYmin = [min(i,j) for i,j in zip(lY,lYmin)]
    lYmax = [max(i,j) for i,j in zip(lY,lYmax)]
    plt.plot(lX,lY,color = 'grey')


plt.plot(lX,lYmax,color='red',lw=2)
plt.plot(lX,lYmin,color='red',lw=2)

plt.savefig('../paper/figs/ex1_UC.png')
plt.savefig('figs/ex1_UC.png')


plt.clf()
a,b,c,d = generate_confusion_matrix(test_results,base_predict)
try:
    s = 1/(1+c/a)
except:
    s = None
try:    
    t = 1/(1+b/d)
except:
    t = None

print('BASE\na=%s\tb=%s\nc=%s\td=%s\ns=%s\tt=%s' %(a,b,c,d,s,t))


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

# print('p = %s'%((a+c)/(a+b+c+d)))

### ROC CURVE
s,fpr = ROC(model = base, data = test_data, results = test_results)
s_lb, fpr_lb, s_ub, fpr_ub = UQ_ROC(models = models, data = test_data, results = test_results)

plt.step(fpr,s,'k')

plt.ylabel('$s$')
plt.xlabel('1-$t$')
plt.plot([0,1],[0,1],'k:')
plt.savefig('../paper/figs/ex1_ROC.png')
plt.savefig('figs/ex1_ROC.png')


plt.step(fpr_lb,s_lb,'b')
plt.step(fpr_ub,s_ub,'r') 
plt.savefig('../paper/figs/ex1_UC_ROC.png')
plt.savefig('figs/ex1_UC_ROC.png')


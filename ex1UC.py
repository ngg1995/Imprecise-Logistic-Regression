import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
import itertools as it
from tqdm import tqdm
import pba
import tikzplotlib

def generate_results(data):

    results = pd.Series(index = data.index, dtype = 'bool')
    for row in data.index:

        results[row] = sum(data.loc[row]) >= len(data.columns)*(15+3*np.random.randn())
    
    return results

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

def uc_logistic_regression(data,result,uncertain):

    models = {}

    for N,i in enumerate(it.product([0,1],repeat=len(uncertain))):

        new_data = pd.concat((data,uncertain), ignore_index = True)
        new_result = pd.concat((results, pd.Series(i)), ignore_index = True)

        model = LogisticRegression()       
        models[str(i)] = model.fit(new_data.to_numpy(),new_result.to_numpy())
        
    return models

# set seed to ensure same data
np.random.seed(10)

# Params
many = 25
dim = 1
few = 1
some = 100

# Generate data
data = pd.DataFrame(30*np.random.rand(many,dim))
results = generate_results(data)

# Generate uncertain points
uncertain = 12.5+pd.DataFrame(5*np.random.rand(few,dim))

# Generate test data 
np.random.seed(111)
test_data = pd.DataFrame(30*np.random.rand(some,dim))
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

# Plot results
plt.scatter(data,results,color='blue')
plt.xlabel('X')
plt.ylabel('$\Pr(X=x)$')

lX = np.linspace(data.min(),data.max(),100)
lY = base.predict_proba(lX.reshape(-1, 1))[:,1]
plt.plot(lX,lY,color='k',zorder=10,lw=2)
tikzplotlib.save('paper/figs/UC1D.tikz')

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


tikzplotlib.save('paper/figs/ex1UC.tikz')

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

print('p = %s'%((a+c)/(a+b+c+d)))
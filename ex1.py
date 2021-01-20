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


def ROC(model, data, results):
    
    s = []
    fpr = []
    
    predictions = model.predict_proba(data)[:,1]
    
    for p in np.linspace(0,1,101):
        a = 0
        b = 0
        c = 0
        d = 0

        for prediction, result in zip(predictions,results):

            if prediction >= p:
                if result:
                    # true positive
                    a += 1
                else:
                    # false positive
                    b+= 1
            else: 
                if result:
                    # false negative
                    c += 1
                else:
                    # true negative
                    d += 1
                    
        
        s.append(a/(a+c))
        fpr.append(b/(b+d))
    return s,fpr
            
            
# set seed to ensure same data
np.random.seed(10)

# Params
many = 2500
dim = 1
few = 500
some = 100

# Generate data
data = pd.DataFrame(30*np.random.rand(many,dim))
results = generate_results(data)
with open('ex1data.csv','w') as f:
    print('x,results', file=f)
    for i in data.index:
        print('%.3f ,%i'%(data.iloc[i],results.iloc[i]),file=f)

# Generate test data 
np.random.seed(111)
test_data = pd.DataFrame(30*np.random.rand(some,dim))
test_results = generate_results(test_data)
# with open('paper/figs/ex1test.csv','w') as f:
#     print('x,result,x,result,x,result,x,result', file=f)
#     for i in data.index:
#         print('%.3f ,%s,%.3f ,%s,%.3f ,%s,%.3f ,%s'%(test_data.iloc[i],test_results.iloc[i],test_data.iloc[i+25],test_results.iloc[i+25],test_data.iloc[i+50],test_results.iloc[i+50],test_data.iloc[i+75],test_results.iloc[i+75]),file=f)
# Fit base model
base = LogisticRegression()
base.fit(data.to_numpy(),results.to_numpy())

# Get beta values
print('b0 = %.3f, b1 = %.3f'%(base.intercept_,base.coef_))

# Classify test data
base_predict = base.predict(test_data)

# Plot results
plt.scatter(data,results,color='blue')
plt.xlabel('X')
plt.ylabel('$\Pr(X=1)$')

lX = np.linspace(data.min(),data.max(),100)
lY = base.predict_proba(lX.reshape(-1, 1))[:,1]
plt.plot(lX,lY,color='k',zorder=10,lw=2)
# tikzplotlib.save('paper/figs/ex1.tikz')

plt.clf()

### ROC CURVE
s,fpr = ROC(base,test_data,test_results)
plt.plot(fpr,s,'r')
plt.plot([0,0],[1,1],'k:')
plt.xlabel('1-$t$')
plt.ylabel('$s$')
# tikzplotlib.save('paper/figs/ex1_ROC.tikz')
# Get confusion matrix

# a,b,c,d = generate_confusion_matrix(test_results,base_predict)
# try:
#     s = 1/(1+c/a)
# except:
#     s = None
# try:    
#     t = 1/(1+b/d)
# except:
#     t = None

# print('BASE\na=%s\tb=%s\nc=%s\td=%s\ns=%s\tt=%s' %(a,b,c,d,s,t))


# aa,bb,cc,dd = generate_confusion_matrix(test_results,predictions)
# try:
#     ss = 1/(1+cc/aa)
# except:
#     ss = None
# try:    
#     tt = 1/(1+bb/dd)
# except:
#     tt = None
# print('KEEP\na=%s\tb=%s\nc=%s\td=%s\ns=%s\tt=%s' %(aa,bb,cc,dd,ss,tt))


# aaa,bbb,ccc,ddd = generate_confusion_matrix(test_results,predictions,throw = True)
# try:
#     sss = 1/(1+ccc/aaa)
# except:
#     sss = None
# try:    
#     ttt = 1/(1+bbb/ddd)
# except:
#     ttt = None
# print('THROW\na=%s\tb=%s\nc=%s\td=%s\ns=%s\tt=%s' %(aaa,bbb,ccc,ddd,sss,ttt))

# print('p = %s'%((a+c)/(a+b+c+d)))
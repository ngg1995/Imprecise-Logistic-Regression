import pandas as pd
from matplotlib import pyplot as plt
import itertools as it
import random
from sklearn.linear_model import LogisticRegression

from LRF import *

iris_data = pd.read_csv('iris.csv',index_col = None)

features = iris_data[['sepal length','sepal width','petal length', 'petal width']]
results = iris_data['species'] == 'Iris-versicolor'

test_data, test_results, train_data, train_results, uq_data = split_data(features, results, test_frac = 0.5, uq_frac = 0.1)

l = len(features.columns)
colors = ['b' if d else 'r'for c,d in results.iteritems()]
for i,(j,k) in enumerate(it.product(features.columns,repeat=2)):
    if j != k:
        plt.subplot(l,l,i+1)
        plt.scatter(features[j],features[k],c=colors)
        plt.scatter(uq_data[j],uq_data[k],c='k')
    else:
        ax = plt.subplot(l,l,i+1, frameon=False, xlim = [0,1], ylim = [0,1],yticks = [], xticks = [])
        ax.text(.5,.5,s = j, fontsize = 'large', ha = 'center')
        
plt.savefig('figs/iris_grid.png')

# Fit model
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
a,b,c,d = generate_confusion_matrix(base_predict,test_results)
print('TP=%i\tFP=%i\nFN=%i\tTN=%i' %(a,b,c,d))

# Calculate sensitivity and specificity
print('Sensitivity = %.2f' %(a/(a+c)))
print('Specificity = %.2f' %(d/(b+d)))

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

plt.clf()
### ROC CURVE
s,fpr = ROC(model = base, data = test_data, results = test_results)
s_lb, fpr_lb, s_ub, fpr_ub = UQ_ROC(models = uq_models, data = test_data, results = test_results)

plt.plot(fpr,s,'r')
plt.plot([0,1],[0,1],'k:')
plt.xlabel('1-$t$')
plt.ylabel('$s$')
plt.savefig('figs/iris_ROC.png')

plt.plot(fpr_lb,s_lb,'g')
plt.plot(fpr_ub,s_ub,'k')
plt.savefig('figs/iris_UQ_ROC.png')
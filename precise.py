import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
import itertools as it
from tqdm import tqdm
import pba
import random
from LRF import *
import tikzplotlib
# import matplotlib
# font = {'size'   : 14,'family' : 'Times New Roman'}
# matplotlib.rc('font', **font)

# colors
col_precise = 'black'
col_points = '#A69888'
col_ilr = '#4169E1'
col_ilr2 = '#5d2e46'
col_ilr3 = '#F7AEF8'
col_ilr4 = '#132E32'
col_mid = '#DC143C'

# load dataset
from dataset import train_data, train_results, test_data, test_results

### Fit logistic regression model
base = LogisticRegression()
base.fit(train_data.to_numpy(),train_results.to_numpy())

### Plot results
steps = 300
lX = np.linspace(0,10,steps)
lY = base.predict_proba(lX.reshape(-1, 1))[:,1]

plt.xlabel('$x$')
plt.ylabel('$\pi(x)$')
rng1 = np.random.default_rng(1)
scat_results = [r+0.1*rng1.random() if r else r-0.1*rng1.random() for r in train_results.to_numpy()]
plt.scatter(train_data,scat_results,color=col_points,zorder=10)
# plt.scatter(test_data,test_results,color='green',zorder=10)
plt.plot(lX,lY,color=col_precise,zorder=10,lw=2)
# plt.savefig('figs/precise.png',dpi = 600)
# plt.savefig('../LR-paper/figs/precise.png',dpi = 600)
tikzplotlib.save('figs/precise.tikz',externalize_tables = True, override_externals = True,tex_relative_path_to_data = 'dat/precise/')


### Get confusion matrix 
base_predict = base.predict(test_data)
print(len(test_results))
with open('runinfo/precise_cm.out','w') as f:
    a,b,c,d = generate_confusion_matrix(test_results,base_predict)
    print('TP=%i\tFP=%i\nFN=%i\tTN=%i' %(a,b,c,d),file = f)

    # Calculate sensitivity and specificity
    print('Sensitivity = %.3f' %(a/(a+c)),file = f)
    print('Specificity = %.3f' %(d/(b+d)),file = f)


### ROC/AUC
s,fpr, predictions = ROC(model = base, data = test_data, results = test_results)

rocfig,axroc = plt.subplots(1)
densfig,axdens = plt.subplots(2,1)
dat0 = ['x y']
dat1 = ['x y']
for i,(p,r) in enumerate(zip(predictions,test_results.to_list())):
    yd = np.random.uniform(-0.1,0.1)
    if r:
        dat1 += [f"{p} {yd}"]
        axdens[0].scatter(p,yd,color = 'k',marker = 'o',alpha = 0.5)
    else:
        dat0 += [f"{p} {yd}"]
        axdens[1].scatter(p,yd,color = 'k',marker = 'o',alpha = 0.5)

axroc.plot([0,1],[0,1],linestyle = ':',color=col_points,label = 'Random Classifier')
axroc.set(xlabel = '$fpr$',ylabel='$s$')
axroc.plot(fpr,s,color = col_precise, label = '$\mathcal{LR}(D)$')
axroc.legend()
# rocfig.savefig('figs/precise_ROC.png',dpi = 600)
# rocfig.savefig('../LR-paper/figs/precise_ROC.png',dpi = 600)
# densfig.savefig('figs/precise_dens.png',dpi = 600)
# densfig.savefig('../LR-paper/figs/precise_dens.png',dpi = 600)

tikzplotlib.save('figs/precise_ROC.tikz',figure = rocfig,externalize_tables = True, override_externals = True,tex_relative_path_to_data = 'dat/precise')
# tikzplotlib.save('figs/precise_dens.tikz',figure = densfig,externalize_tables = True, override_externals = True,tex_relative_path_to_data = 'dat/')
print(*dat0,sep='\n',file = open('figs/dat/precise/precise_dens-000.dat','w'))
print(*dat1,sep='\n',file = open('figs/dat/precise/precise_dens-001.dat','w'))

with open('runinfo/precise_auc.out','w') as f:
    print('AUC: %.4f' %auc(s,fpr), file = f)


# ### Hosmer-Lemeshow
# hl, pval = hosmer_lemeshow_test(base,train_data,train_results,g = 10)
# with open('runinfo/precise_HL.out','w') as f:
#     print('hl = %.3f, p = %.5f' %(hl,pval),file = f) 

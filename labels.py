# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
import itertools as it
import tikzplotlib
from tqdm import tqdm
import pba
import random
from sklearn.semi_supervised import SelfTrainingClassifier
# import matplotlib
# font = {'size'   : 14,'family' : 'Times New Roman'}
# matplotlib.rc('font', **font)

# colors
col_precise = 'black'
col_points = '#A69888'
col_ilr = '#4169E1'
col_ilr2 = '#5d2e46'
col_ilr3 = '#FF8C00'
col_ilr4 = '#008000'
col_mid = '#DC143C'

from ImpLogReg import *
from LRF import *

from dataset import *

# %%
### Intervalise data
# drop some results
few = 5 #uncertain points
random.seed(12345) # for reproducability
uq_data_index = random.sample([i for i in train_data.index if abs(train_data.loc[i,0]-5) <= 2], k = few) # clustered around center

uq_data = train_data.loc[uq_data_index]
uq_results = pd.Series([int(train_results.loc[i]) if i not in uq_data_index else pba.I(0,1) for i in train_results.index], index = train_data.index, dtype='O')

# %% [markdown]
### Fit logistic regression model on full dataset
base = LogisticRegression(max_iter=1000)
base.fit(train_data.to_numpy(),train_results.to_numpy())

# %% [markdown]
### Fit models with no uq data
nuq_data = train_data.loc[[i for i in train_data.index if i not in uq_data_index]]
nuq_results = train_results.loc[[i for i in train_data.index if i not in uq_data_index]]
nuq = LogisticRegression(max_iter=1000)
nuq.fit(nuq_data.to_numpy(),nuq_results.to_numpy())

#%%
sslr_results = pd.Series([int(train_results.loc[i]) if i not in uq_data_index else -1 for i in train_results.index], index = train_data.index, dtype=int)
### Fit Semi-supervised logsitic regression
sslr = SelfTrainingClassifier(LogisticRegression())
sslr.fit(train_data,sslr_results)


# %% [markdown]
### Fit UQ models
ilr = ImpLogReg(uncertain_class=True, max_iter = 1000)
ilr.fit(train_data,uq_results)

# %% 
### Plot results
steps = 300
lX = np.linspace(0,10,steps)
lY = base.predict_proba(lX.reshape(-1, 1))[:,1]
lYn = nuq.predict_proba(lX.reshape(-1, 1))[:,1]
lYs = sslr.predict_proba(lX.reshape(-1, 1))[:,1]
lYu = ilr.predict_proba(lX.reshape(-1,1))[:,1]

fig1, ax1 = plt.subplots()

ax1.set_xlabel('$x$')
ax1.set_ylabel('$\pi(x)$')
ax1.plot(lX,lY,color=col_precise,zorder=10,lw=2,label = '$\mathcal{LR}(D)$') 
ax1.plot(lX,lYn,color=col_mid,zorder=10,lw=2,label = '$\mathcal{LR}(F_\\times)$') 
ax1.plot(lX,lYs,color=col_ilr4,zorder=10,lw=2,label = r'$ss$') 
ax1.scatter(nuq_data,nuq_results,color=col_points,zorder=10)
for i in uq_data_index:

    ax1.plot([uq_data.loc[i],uq_data.loc[i]],[0,1],color=col_points)
    # plt.scatter(uq_data.loc[i],train_results.loc[i],marker = 'd',color = 'grey',zorder = 14)

ax1.plot(lX,[i.left for i in lYu],color=col_ilr,lw=2)
ax1.plot(lX,[i.right for i in lYu],color=col_ilr,lw=2,label = '$\mathcal{ILR}(F)$')
# fig1.savefig('../LR-paper/figs/labels.png',dpi = 600)
# fig1.savefig('figs/labels.png',dpi = 600)
ax1.legend()
tikzplotlib.save('figs/labels.tikz',figure = fig1,externalize_tables = True, override_externals = True,tex_relative_path_to_data = 'dat/labels/')

# %% [markdown]
### Get confusion matrix
# Classify test data
base_predict = base.predict(test_data)

# CLASSIFY NO_UQ MODEL DATA 
nuq_predict = nuq.predict(test_data)

# CLASSIFY NO_UQ MODEL DATA 
sslr_predict = sslr.predict(test_data)

# CLASSIFY UQ MODEL 
ilr_predict = ilr.predict(test_data)

with open('runinfo/labels_cm.out','w') as f:
    print('TRUE MODEL',file = f)
    a,b,c,d = generate_confusion_matrix(test_results,base_predict)
    print('TP=%i\tFP=%i\nFN=%i\tTN=%i' %(a,b,c,d),file = f)

    # Calculate sensitivity and specificity
    print('Sensitivity = %.3f' %(a/(a+c)),file = f)
    print('Specificity = %.3f' %(d/(b+d)),file = f)

    print('\nDISCARDED DATA MODEL',file = f)
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
    
    
    print('\nSSLR MODEL',file = f)
    aa,bb,cc,dd = generate_confusion_matrix(test_results,sslr_predict)
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
    
    print('\nUQ MODEL',file = f)
    
    aaai,bbbi,ccci,dddi = generate_confusion_matrix(test_results,ilr_predict,throw = False)
    try:
        sssi = aaai/(a+c)
    except:
        sssi = None
    try:    
        ttti = dddi/(b+d)
    except:
        ttti = None

    print('TP=[%i,%i]\tFP=[%i,%i]\nFN=[%i,%i]\tTN=[%i,%i]' %(*aaai,*bbbi,*ccci,*dddi),file = f)

    # Calculate sensitivity and specificity
    print('Sensitivity = [%.3f,%.3f]\nSpecificity = [%.3f,%.3f]\n' %(*sssi,*ttti),file = f)

    
    aaa,bbb,ccc,ddd,eee,fff = generate_confusion_matrix(test_results,ilr_predict,throw = True)
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
    print('sigma = %.3f' %(eee/(aaa+ccc+eee)),file = f)
    print('tau = %.3f' %(fff/(bbb+ddd+fff)),file = f)
   
# %% [markdown]
s,fpr,probabilities = ROC(model = base, data = test_data, results = test_results)
nuq_s,nuq_fpr,nuq_probabilities = ROC(model = nuq, data = test_data, results = test_results)
s_t, fpr_t, Sigma, Tau = incert_ROC(ilr, test_data, test_results)

s_i, fpr_i,ilr_probabilities = ROC(ilr, test_data, test_results)

densfig,axdens = plt.subplots(nrows = 2, sharex= True)

dat1 = ['x y']
dat2 = ['x y']
dat3 = ['x y']
dat4 = ['x y']

for i,(p,u,nuqp,r) in enumerate(zip(probabilities,ilr_probabilities,nuq_probabilities,test_results.to_list())):
    yd = np.random.uniform(-0.1,0.1)
    if r:
        dat1 += [f"{p} {yd}"]
        dat2 += [f"{nuqp} {0.21+yd}"]
        # axdens[0].scatter(p,yd,color = 'k',marker = 'o',alpha = 0.5)
        # axdens[0].scatter(nuqp,0.21+yd,color = col_mid,marker = 'o',alpha = 0.5)
        axdens[0].plot([*u],[yd-0.21,yd-0.21],color = col_ilr, alpha = 0.3)
        axdens[0].scatter([*u],[yd-0.21,yd-0.21],color = col_ilr, marker = '|')
    else:
        dat3 += [f"{p} {yd}"]
        dat4 += [f"{nuqp} {0.21+yd}"]
        # axdens[1].scatter(p,yd,color = 'k',marker = 'o',alpha = 0.5)
        # axdens[1].scatter(nuqp,0.21+yd,color = col_mid,marker = 'o',alpha = 0.5)
        axdens[1].plot([*u],[yd-0.21,yd-0.21],color = col_ilr, alpha = 0.3)
        axdens[1].scatter([*u],[yd-0.21,yd-0.21],color = col_ilr, marker = '|')
        
        
axdens[0].set(ylabel = '1',yticks = [])
axdens[1].set(xlabel = '$\pi(x)$',ylabel = '0',yticks = [],xlim  = (0, 1))

densfig.tight_layout()

rocfig,axroc = plt.subplots(1,1)

xl = []
xu = []
yl = []
yu = []
for i,j in zip(fpr_i,s_i):
    
    if not isinstance(i,pba.Interval):
        i = pba.I(i)
    if not isinstance(j,pba.Interval):
        j = pba.I(j)
      
    xl.append(i.left  )
    xu.append(i.right  )
    
    yl.append(j.left)
    yu.append(j.right)
    
axroc.plot(xl,yu, col_ilr,label = '$\mathcal{ILR}(F)$')
axroc.plot(xu,yl, col_ilr )
axroc.plot([0,1],[0,1],linestyle = ':',color=col_points)

axroc.set(xlabel = '$fpr$',ylabel='$s$')
axroc.plot(fpr,s,'k',label = '$\mathcal{LR}(D)$')
axroc.plot(nuq_fpr,nuq_s,color=col_mid,linestyle='--',label='$\mathcal{LR}(F_\\times)$')
axroc.plot(fpr_t,s_t,col_ilr2,label='$\mathcal{ILR}(F)$ (Predictive)')
axroc.legend()
# rocfig.savefig('figs/labels_ROC.png',dpi = 600)
# rocfig.savefig('../LR-paper/figs/labels_ROC.png',dpi = 600)
# densfig.savefig('figs/labels_dens.png',dpi =600)
# densfig.savefig('../LR-paper/figs/labels_dens.png',dpi =600)

tikzplotlib.save('figs/labels_ROC.tikz',figure = rocfig,externalize_tables = True, override_externals = True,tex_relative_path_to_data = 'dat/labels/')

tikzplotlib.save('figs/labels_dens.tikz',figure = densfig,externalize_tables = False, override_externals = True,tex_relative_path_to_data = 'dat/labels/')
print(*dat1,sep='\n',file = open('figs/dat/labels/labels_dens-000.dat','w'))
print(*dat2,sep='\n',file = open('figs/dat/labels/labels_dens-001.dat','w'))
print(*dat3,sep='\n',file = open('figs/dat/labels/labels_dens-002.dat','w'))
print(*dat4,sep='\n',file = open('figs/dat/labels/labels_dens-003.dat','w'))

#%%
with open('runinfo/labels_auc.out','w') as f:
    print('NO UNCERTAINTY: %.4f' %auc(s,fpr), file = f)
    print('MIDPOINTS: %.4F' %auc(nuq_s,nuq_fpr),file = f)
    print('THROW: %.4f' %auc(s_t,fpr_t), file = f)
    print('ILR: [%.3f,%.3f]'  %(auc(yl,xu),auc(yu,xl)), file = f)
    
    # print('INTERVALS: [%.3f,%.3f]' %(auc_int_min,auc_int_max), file = f)
    


fig2,ax2 = plt.subplots()
ax2 = plt.axes(projection='3d',elev = 45,azim = -45,proj_type = 'ortho')
ax2.set_xlabel('$fpr$')
ax2.set_ylabel('$s$')
# ax2.set_zlabel('$1-\sigma,1-\\tau$')
ax2.plot(fpr_t,s_t,col_ilr2,alpha = 0.5)
ax2.plot3D(fpr_t,s_t,Sigma,col_ilr3,label = '$\\sigma$')
ax2.plot3D(fpr_t,s_t,Tau,col_ilr4,label = '$\\tau$')


ax2.legend()

# fig2.savefig('figs/labels_ROC3D.png',dpi = 600)
# fig2.savefig('../LR-paper/figs/labels_ROC3D.png',dpi = 600)

tikzplotlib.save("figs/labels_ROC3D.tikz",figure = fig2,externalize_tables = True, override_externals = True,tex_relative_path_to_data = 'dat/labels/')

fig3, ax3 = plt.subplots()

ax3.set_xlabel('$fpr$/$s$')
ax3.set_ylabel('$\\sigma$/$\\tau$')
ax3.plot(s_t,Sigma,col_ilr3,label = '$\\sigma$ v $s$')
ax3.plot(fpr_t,Tau,col_ilr4,label = '$\\tau$ v $fpr$')
ax3.legend()


# fig3.savefig('figs/labels_ST.png',dpi = 600)
# fig3.savefig('../LR-paper/figs/labels_ST.png',dpi = 600)
tikzplotlib.save("figs/labels_ST.tikz",figure = fig3,externalize_tables = True, override_externals = True,tex_relative_path_to_data = 'dat/labels/')


# %% [markdown]
# ### Hosmer-Lemeshow

# hl_b, pval_b = hosmer_lemeshow_test(base,train_data,train_results,g = 10)

# hl_nuq, pval_nuq = hosmer_lemeshow_test(nuq,train_data,train_results,g = 10)
# #
# hl_uq, pval_uq = UQ_hosmer_lemeshow_test(ilr,train_data,train_results,g = 10)

# with open('runinfo/labels_HL.out','w') as f:
#     print('base\nhl = %.3f, p = %.3f' %(hl_b,pval_b),file = f)
#     print('no UQ\nhl = %.3f, p = %.3f' %(hl_nuq,pval_nuq),file = f) 

#     print('UQ\nhl = [%.3f,%.3f], p = [%.3f,%.3f]' %(*hl_uq,*pval_uq),file = f) 

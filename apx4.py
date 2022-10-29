#%%
from multiprocessing import freeze_support
import multiprocessing
from turtle import position
import pba
import numpy as np
import tikzplotlib
from sklearn.linear_model import LogisticRegression as LogReg
from matplotlib import pyplot as plt
import pandas as pd
from ImpLogReg import ImpLogReg
from itertools import product
from tqdm import tqdm, trange
#%%
def get_vals_from_intervals(r,int_y):
    return np.array([i*j.width() + j.left for i,j in zip(r,int_y)]).reshape(-1, 1)

def area(xl,yl,yu,nmax,nmin):
    
    a = np.trapz(np.maximum(nmax-yu,np.zeros(yl.size)),xl)
    b = np.trapz(np.maximum(yl-nmin,np.zeros(yl.size)),xl)
    
    d = np.trapz(nmax,xl) - np.trapz(nmin,xl)
    
    return (a+b)/d
#%%
fig1,ax1 = plt.subplots(1,1)
few = 4

y = pd.Series([0,0,1,1])
int_x = [pba.I(1,2),pba.I(1.9,2.9),pba.I(2.75,3.75),pba.I(3.5,4.5)]


ax1.plot((int_x[0].left,int_x[0].right),[-0.02,-0.02])
ax1.plot((int_x[1].left,int_x[1].right),[-0.01,-0.01])
ax1.plot((int_x[2].left,int_x[2].right),[1.01,1.01])
ax1.plot((int_x[3].left,int_x[3].right),[1.02,1.02])

w = [5]*few

ILR = ImpLogReg(uncertain_data=True)
ILR.fit(pd.DataFrame({1:int_x},dtype='O'),y,sample_weight=w)

xl = np.linspace(0,5,1000)

yl = [i.left for i in ILR.predict_proba(xl.reshape(-1, 1))[:,1]]
yr = [i.right for i in ILR.predict_proba(xl.reshape(-1, 1))[:,1]]



nmin = np.full(len(xl),np.inf)
nmax = np.full(len(xl),-np.inf)


for i in tqdm(list(product(np.linspace(0,1,10),repeat=len(int_x)))):

    r = get_vals_from_intervals(i,int_x)
    # r = get_vals_from_intervals(np.random.random(4),int_x)
    R = LogReg().fit(r,y,sample_weight=w)
    yy = R.predict_proba(xl.reshape(-1,1))[:,1]
    # plt.plot(xl,yy,color='lightgrey')
    nmin = np.minimum(nmin.ravel(),yy.ravel())
    nmax = np.maximum(nmax.ravel(),yy.ravel())
    # n += [sum((yl.ravel() < yy.ravel()) & (yu.ravel() > yy.ravel())) != few]
ax1.fill_between(xl,nmin,nmax,color='grey')
ax1.plot(xl,yl,'k')
ax1.plot(xl,yr,'k')
print(f"A = {area(xl,np.array(yl),np.array(yr),nmax,nmin)}")
tikzplotlib.save('figs/apx_log4_all.tikz',figure = fig1,externalize_tables = True, override_externals = True,tex_relative_path_to_data = 'dat/apx/')
# %%
def func(value):
    rng = np.random.default_rng(value)
    few = 4
    xl = np.linspace(0,5,1000)
    y_vals = [
        [0,0,0,1],
        [0,0,1,1],
        [0,1,1,1]
    ]
    w = [5,5,5,5]

    y = rng.choice(y_vals)
    rng.shuffle(y)
    x = rng.uniform(0,4,few)
    int_x = [pba.I(i,i+1) for i in x]
    
    ILR = ImpLogReg(uncertain_data=True)
    ILR.tqdm_disable = True
    ILR.fit(pd.DataFrame({1:int_x},dtype='O'),y,sample_weight=w)
    yl = [i.left for i in ILR.predict_proba(xl.reshape(-1, 1))[:,1]]
    yr = [i.right for i in ILR.predict_proba(xl.reshape(-1, 1))[:,1]]

    nmin = np.full(len(xl),np.inf)
    nmax = np.full(len(xl),-np.inf)

    for i in product(np.linspace(0,1,10),repeat=len(int_x)):

        r = get_vals_from_intervals(i,int_x)
        R = LogReg().fit(r,y,sample_weight=w)
        yy = R.predict_proba(xl.reshape(-1,1))[:,1]
        nmin = np.minimum(nmin.ravel(),yy.ravel())
        nmax = np.maximum(nmax.ravel(),yy.ravel())
        
    return  area(xl,np.array(yl),np.array(yr),nmax,nmin)

if __name__ == '__main__':
    freeze_support()
    many = 10000
    pool_obj = multiprocessing.Pool()
    areas = tqdm(pool_obj.imap(func,range(many)),total = many)
    fig_hist,ax_hist = plt.subplots(1,1)
    ax_hist.hist(areas,bins=[-0.001,0.001,0.0125,0.025,0.0375,0.05,0.0625,0.075,0.0875,0.1,0.1125,0.125,0.1375,0.15,0.1625,0.175,.1875,0.2,1])
    tikzplotlib.save("figs/apx_log4_hist.tikz",figure=fig_hist)
# %%

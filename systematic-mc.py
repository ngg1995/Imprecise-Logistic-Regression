#%%
from multiprocessing import freeze_support
import multiprocessing
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
few = 6
rng = np.random.default_rng(111)
# x = rng.uniform(0.5,4.5,few)
# y = x > rng.normal(loc = 2.5,scale = 0.75,size=few) 
int_x = [pba.I(0.5,1.5),pba.I(1.25,2.25),pba.I(2,3),pba.I(2.75,3.75),pba.I(3.1,4.1),pba.I(3.8,4.8)]
y = [0,0,0,1,1,1]
fig,ax = plt.subplots(1,1)
for i,j in zip(int_x,y):
    if j:
        jt = rng.uniform(0,0.1)
    else:
        jt = -rng.uniform(0,0.1)
        
    ax.plot([i.left,i.right],[j+jt,j+jt])
# %%
### Monte Carlo
def monte_carlo(int_x,y):
    rng2 = np.random.default_rng(2)
    many = 10000

    xl = np.linspace(0,5,1000)

    mc_min = np.full(len(xl),np.inf)
    mc_max = np.full(len(xl),-np.inf)

    for i in trange(many):
        X = get_vals_from_intervals(rng2.random(len(int_x)),int_x)
        LR = LogReg()
        LR.fit(X,y)
        yy = LR.predict_proba(xl.reshape(-1,1))[:,1]

        mc_min = np.minimum(mc_min.ravel(),yy.ravel())
        mc_max = np.maximum(mc_max.ravel(),yy.ravel())

    return mc_min, mc_max
# %%
### Systematic
def systematic(i,int_x=int_x,y=y):
    r = product(np.linspace(0,1,10),repeat=len(int_x))
    for _ in range(i+1):
        a = next(r)
    xl = np.linspace(0,5,1000)
    X = get_vals_from_intervals(a,int_x)
    LR = LogReg()
    LR.fit(X,y)
    yy = LR.predict_proba(xl.reshape(-1,1))[:,1]

    return yy

# %%
if __name__ == '__main__':
    xl = np.linspace(0,5,1000)
    freeze_support()
    pool_obj = multiprocessing.Pool(5)

    sc_min = np.full(len(xl),np.inf)
    sc_max = np.full(len(xl),-np.inf)
    

    YY = pool_obj.map(systematic,trange(10**6))

    for Y in tqdm(YY):
        sc_min = np.minimum(sc_min.ravel(),Y.ravel())
        sc_max = np.maximum(sc_max.ravel(),Y.ravel())
    
    mc_min, mc_max =  monte_carlo(int_x,y)
    fig,ax = plt.subplots(1,1)
    for i,j in zip(int_x,y):
        if j:
            jt = rng.uniform(0,0.1)
        else:
            jt = -rng.uniform(0,0.1)
            
        ax.plot([i.left,i.right],[j+jt,j+jt])
    ax.plot(xl,mc_min,'r')
    ax.plot(xl,mc_max,'r')

    ax.plot(xl,sc_min,'grey')
    ax.plot(xl,sc_max,'grey')
    tikzplotlib.save('figs/feature-mc.tikz',figure = fig, externalize_tables = True, override_externals = True,tex_relative_path_to_data = 'dat/features/')
    pool_obj.close()
    print(f"MC - {area(xl,mc_min,mc_max,sc_max,sc_min)}")
    iX = pd.DataFrame({1:int_x},dtype = 'O')
    iY = pd.Series(y)
    ilr = ImpLogReg(uncertain_data=True)
    ilr.fit(iX,y)
    fig2,ax2 = plt.subplots(1,1)
    for i,j in zip(int_x,y):
        if j:
            jt = rng.uniform(0,0.1)
        else:
            jt = -rng.uniform(0,0.1)
            
        ax2.plot([i.left,i.right],[j+jt,j+jt])
        
    for l,m in ilr.models.items():
        lY = m.predict_proba(xl.reshape(-1, 1))[:,1]
        ax2.plot(xl,lY, linewidth = 2,label = l)
    ax2.legend()
    
    
    tikzplotlib.save('figs/feature-minmax.tikz',figure = fig2, externalize_tables = True, override_externals = True,tex_relative_path_to_data = 'dat/features/')
    yl = np.array([i.left for i in ilr.predict_proba(xl.reshape(-1, 1))[:,1]])
    yr = np.array([i.right for i in ilr.predict_proba(xl.reshape(-1, 1))[:,1]])
    print(f"F3 - {area(xl,yl,yr,sc_max,sc_min)}")

    
    iX = pd.DataFrame({1:int_x},dtype = 'O')
    ilr = ImpLogReg(uncertain_data=True)
    ilr.fit(iX,iY,fast=True,n_p_vals=100)
    fig3,ax3 = plt.subplots(1,1)
    for i,j in zip(int_x,y):
        if j:
            jt = rng.uniform(0,0.1)
        else:
            jt = -rng.uniform(0,0.1)
            
        ax3.plot([i.left,i.right],[j+jt,j+jt])

    yl = np.array([i.left for i in ilr.predict_proba(xl.reshape(-1, 1))[:,1]])
    yr = np.array([i.right for i in ilr.predict_proba(xl.reshape(-1, 1))[:,1]])
    ax3.plot(xl,yl)
    ax3.plot(xl,yr)
    print(f"F4 - {area(xl,yl,yr,sc_max,sc_min)}")
    
    tikzplotlib.save('figs/feature-6line.tikz',figure = fig3, externalize_tables = True, override_externals = True,tex_relative_path_to_data = 'dat/features/')


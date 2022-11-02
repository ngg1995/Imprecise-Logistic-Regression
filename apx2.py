#%%
import pba
import numpy as np
import tikzplotlib
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from ImpLogReg import ImpLogReg
#%%
def fit_lr(x1,x2,y1,y2):
    X = [x1] + [x2]
    Y = [y1] + [y2]
    lr = LogisticRegression(tol=1e-15)
    lr.fit(np.array(X).reshape(-1,1),Y,sample_weight=[1000,1000])
    return lr
#%%
x1 = pba.I(1,2)
x2 = pba.I(3,4)
y1 = 0
y2 = 1

for i in x1: 
    for j in x2:
        lr = fit_lr(i,j,y1,y2)
        print(f"\\addplot[ultra thick,color=col1,domain=-1:6] {{1/(1+exp(-({lr.coef_[0][0]:.3f}*x+{lr.intercept_[0]:.3f})))}};")

        plt.plot([*x1],[y1,y1],'b')
        plt.plot([*x2],[y2,y2],'r')
        plt.plot(np.linspace(0,5,100),lr.predict_proba(np.linspace(0,5,100).reshape(-1,1))[:,1])
# %%

rng = np.random.default_rng(1)

for i in np.linspace(0,1,11):
    for j in np.linspace(0,1,11):
        
        y1 = 0
        y2 = 1

        lr = fit_lr(x1.left + i,x2.left + j,y1,y2)

        print(f"\\addplot[ultra thick,color=points,domain=-1:6,samples=100] {{1/(1+exp(-({lr.coef_[0][0]:.3f}*x+{lr.intercept_[0]:.3f})))}};")

# %%
x1 = pba.I(1,2.45)
x2 = pba.I(2.55,4)
y1 = 0
y2 = 1

for i in x1: 
    for j in x2:
        lr = fit_lr(i,j,y1,y2)
        print(f"\\addplot[ultra thick,color=col1,domain=-1:6] {{1/(1+exp(-({lr.coef_[0][0]:.3f}*x+{lr.intercept_[0]:.3f})))}};")

        plt.plot([*x1],[y1,y1],'b')
        plt.plot([*x2],[y2,y2],'r')
        plt.plot(np.linspace(0,5,100),lr.predict_proba(np.linspace(0,5,100).reshape(-1,1))[:,1])
# %%

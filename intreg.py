import pba
import random
import numpy as np
import matplotlib.pyplot as plt 

# seed for reproducability
random.seed(0)

# set line
Y = lambda x: 3*x+4

# generate many uncertain data points
many = 10
x = [10*random.random() for i in range(many)]
y = [pba.I(Y(i)-(1+u),Y(i)+(1-u)) for i,u in zip(x,[random.gauss(0,0.5) for i in range(many)])]

# plot
for i,j in zip(x,y):
    plt.plot([i,i],[j.lo(),j.hi()])
    
# plt.plot(np.linspace(0,10,1000),Y(np.linspace(0,10,1000)))
plt.show()
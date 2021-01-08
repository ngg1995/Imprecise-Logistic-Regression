import numpy as np
import itertools
from tqdm import tqdm
from scipy import optimize
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
import pba

def int_exp(i):
    if i.__class__.__name__ == 'Interval':
        lo = np.exp(i.Left)
        hi = np.exp(i.Right)
        return pba.Interval(lo,hi)
    else:
        return np.exp(i)

# Method to make predictions
def predict(X, b0, b1):
    return np.array([1 / (1 + int_exp(-1*b0 + -1*b1*x)) for x in X])

def logistic_regression(X, Y):

    # Initializing variables
    b0 = 0
    b1 = 0
    L = 0.001
    epochs = 1000

    for epoch in range(epochs):
        y_pred = predict(X, b0, b1)
        D_b0 = -2 * sum((Y - y_pred) * y_pred * (1 - y_pred))  # Derivative of loss wrt b0
        D_b1 = -2 * sum(X * (Y - y_pred) * y_pred * (1 - y_pred))  # Derivative of loss wrt b1
        # Update b0 and b1
        b0 = b0 - L * D_b0
        b1 = b1 - L * D_b1
    
    return b0, b1

# set seed to ensure same data
np.random.seed(11)

# Params
many = 5
dim = 1
few = 5
some = 10
eps = 0.05

XX = np.linspace(0,3,100)
data = 3*np.random.rand(many)
results = [i > 1.5+np.random.rand()/3 for i in data]
b0,b1 = logistic_regression(data,results)
plt.plot(XX,predict(XX,b0,b1))

B0 = []
B1 = []

for p in tqdm(itertools.product([-eps,eps],repeat=many)):

    data1 = data + p

    b0,b1 = logistic_regression(data1,results)
    B0.append(b0)
    B1.append(b1)


for d,r in zip(data1,results):
    plt.plot([d-eps,d+eps],[r,r],color='c',marker='|')


plt.show()
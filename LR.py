import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression


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
np.random.seed(1)

# Generate data
many = 24
some = 600
sample_X = np.sort(np.random.randn(many,1))
sample_y = sample_X > np.random.rand(many,1)
test_x = np.sort(np.random.randn(some,1))
test_y = test_x > np.random.rand(some,1)

# Fit model
model = LogisticRegression()
model.fit(sample_X,sample_y)

# Classify test data
test_predict = model.predict(test_x)


# Plot results
# plt.scatter(sample_X,sample_y)
# plt.scatter(test_x,test_y,marker = 'o')
# plt.scatter(test_x,test_predict,marker = 'x')
# lX = np.linspace(min(sample_X),max(sample_X),1001)
# lY = model.predict_proba(np.linspace(min(sample_X),max(sample_X),1001).reshape(-1, 1))[:,1]
# plt.plot(lX,lY,color='k')
# plt.show()

s,nt = ROC(model,test_x,test_y) 
plt.plot(nt,s)
plt.show()
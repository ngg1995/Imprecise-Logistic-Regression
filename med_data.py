import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def generate_confusion_matrix(predictions,results):
    
    TP = 0
    FP = 0
    FN = 0
    TN = 0

    for prediction, result in zip(predictions,results):
        # the zip() function allows us to iterate over both lists at the same time
        
        if prediction and result:
            TP += 1
        elif prediction and not result:
            FP += 1
        elif not prediction and result:
            FN += 1
        else:
            TN += 1
            
    return TP,FP,FN,TN


# Import the data
heart_data = pd.read_csv('SAheart.csv',index_col = 'patient')

# Split the data into risk factors and result
factors = heart_data[['sbp','tobacco','ldl','adiposity','famhist','typea','obesity','alcohol','age']]
chd = heart_data['chd']

# Randomly split the data into sampels that can be tested on and trained with
factors_train, factors_test, chd_train, chd_test = train_test_split(factors,chd,train_size=0.333,random_state=1)

# Fit model
model = LogisticRegression(max_iter=200)
model.fit(factors_train,chd_train)

# Make preictions from test data
test_predict = model.predict(factors_test)

# Get confusion matrix
a,b,c,d = generate_confusion_matrix(test_predict,chd_test)
print('TP=%i\tFP=%i\nFN=%i\tTN=%i' %(a,b,c,d))

# Calculate sensitivity and specificity
print('Sensitivity = %.2f' %(a/(a+c)))
print('Specificity = %.2f' %(d/(b+d)))



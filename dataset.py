
import numpy as np
import pandas as pd

def generate_results(data):
    rng_local = np.random.default_rng(1)
    results = pd.Series(index = data.index, dtype = 'bool')
    
    for i in data.index:
        if data.loc[i,0] < 3:
            results.loc[i] = False
        elif data.loc[i,0] > 7:
            results.loc[i] = True
        else:
            results.loc[i] = rng_local.random() < 0.5
            
    return results


### Generate Data
# set seed for reproducability
np.random.seed(10)

# Params
some = 50 # training datapoints
many = 100 # many test samples

train_data = pd.DataFrame(10*np.random.rand(some,1))
train_results = generate_results(train_data)

test_data = pd.DataFrame(10*np.random.rand(many,1))
test_results = generate_results(test_data)
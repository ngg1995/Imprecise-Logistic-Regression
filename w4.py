import numpy as np 
import matplotlib.pyplot as plt 
x = np.array(range(1,11))

y = np.array([200,250,400,600,700,750,800,800,900,1100])

plt.plot(x, y, 'bo') 
m = np.linalg.lstsq(x[:,np.newaxis], y, rcond=None) 
plt.plot(x, m[0]*x, 'b-')
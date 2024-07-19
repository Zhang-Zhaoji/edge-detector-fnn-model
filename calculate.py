import numpy as np
import math
a = [0]*90
for i in range(90):
    if i < 73:
        a[i] = np.longdouble(np.power(0.994,i)*0.006)
    else:
        a[i] = np.longdouble(np.power(0.994,i-72)*(0.006+0.994*(i-72)/17)*math.factorial(17)/math.factorial(17-i+72)/np.power(17,i-72))
print(np.sum(a))
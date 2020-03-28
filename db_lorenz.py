#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import numpy as np
from numpy.random import uniform
from scipy.integrate import solve_ivp
import numpy as np


# In[2]:


h = 0.025
ti = 0.
tf = 50.
sig_len = 2000
t_jump = (tf-ti)/(sig_len*h)
if t_jump%1.0 != 0 : print("temporal step problem")
n_traj = 50000
tv = np.arange(ti,tf,h)

def solve_lorenz(rho, sigma=10.0, beta=8./3.):

    def lorenz_deriv(t0, x_y_z, sigma=sigma, beta=beta, rho=rho):
        x, y, z = x_y_z
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]        
    np.random.seed()
    xxx = np.array([uniform(-14.0,14.0),uniform(-22.0,22.0),uniform(7.0,43.0)])
    a = solve_ivp(lorenz_deriv, t_span=(0.,50), y0=xxx, method='RK45', t_eval=tv)
    if a.success == True:
        return a.y
    else:
        print('Error during integration. Exiting.')
        exit()
        #return 1


# In[ ]:

import sys
try :
    r = float(sys.argv[1])
    header = f'r={r:5} '
    print(header, "Starting.")
except : 
    print("Problem parsing args. Exiting."); exit()

db = np.ndarray(shape=(n_traj,3,sig_len))

for j in range(n_traj):
    db[j,:,:] = solve_lorenz(r)
    if j%(n_traj//10)==0: print(header,f'{j:6}/{n_traj:6}')

np.save(f"/scratch/scarpolini/databases/db_lorenz_{r:.1f}",db)
#np.save(f"db_lorenz_{r:.1f}",db)

print(header,'Done!')

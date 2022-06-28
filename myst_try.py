# try mystic

import numpy as np
M = 10
N = 3
Q = 10
C = 10
# lets be lazy, and generate s and u randomly...
s = np.random.randint(-Q,Q, size=(M,N,N))
u = np.random.randint(-Q,Q, size=(M,N))
def percentile(p, x):
    x = np.sort(x)
    p = 0.01 * p * len(x)
    if int(p) != p:        
        return x[int(np.floor(p))]
    p = int(p)    
    return x[p:p+2].mean()

def objective(x, p=5): 
    # inverted objective, to find the max
    return -1*percentile(p, [np.dot(np.atleast_2d(u[i]), x)[0] for i in range(0,M-1)])

def constraint(x, p=95, v=C): 
    # 95%(xTsx) - v <= 0    
    x = np.atleast_2d(x)
    return percentile(p, [np.dot(np.dot(x,s[i]),x.T)[0,0] for i in range(0,M-1)]) - v

bounds = [(0,1) for i in range(0,N)]


from mystic.penalty import quadratic_inequality
@quadratic_inequality(constraint, k=1e4)
def penalty(x):
    return 0.0

from mystic.solvers import diffev2
from mystic.monitors import VerboseMonitor
mon = VerboseMonitor(10)
result = diffev2(objective, x0=bounds, penalty=penalty, npop=10, gtol=200, disp=False, full_output=True, itermon=mon, maxiter=M*N*100)
print(result[0])
print(result[1])

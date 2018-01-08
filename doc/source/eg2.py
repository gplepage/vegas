import vegas
import numpy as np
from outputsplitter import log_stdout

def f(x):
   return x[0] * x[1] ** 2

m = vegas.AdaptiveMap([[0, 1], [0, 1]], ninc=5) 

ny = 1000
y = np.random.uniform(0., 1., (ny, 2))  # 1000 random y's

x = np.empty(y.shape, float)            # work space
jac = np.empty(y.shape[0], float)
f2 = np.empty(y.shape[0], float)

log_stdout('eg2a.out')
print('initial grid:')
print m.settings()

for itn in range(5):                    # 20 iterations to adapt
   m.map(y, x, jac)                     # compute x's and jac

   for j in range(ny):                  # compute training data
      f2[j] = (jac[j] * f(x[j])) ** 2

   m.add_training_data(y, f2)           # adapt
   m.adapt(alpha=1.5)

   print('iteration %d:' % itn)
   print(m.settings())

import vegas
import math
import gvar as gv
from outputsplitter import log_stdout, unlog_stdout

def f(x):
    dx2 = 0
    for d in range(4):
        dx2 += (x[d] - 0.5) ** 2
    return math.exp(-dx2 * 100.) * 1013.2118364296088

# save
gv.ranseed(1)
log_stdout('eg9a.out')
integ = vegas.Integrator([[-1, 1], [0, 1], [0, 1], [0, 1]])

result = integ(f, nitn=5, neval=1000, save='save.pkl')
print(result.summary())
print('result = %s    Q = %.2f' % (result, result.Q))
unlog_stdout()
print(integ.settings())
import pickle
with open('save.pkl', 'rb') as ifile:
    result = pickle.load(ifile)
print(result.summary())
print('result = %s    Q = %.2f' % (result, result.Q))

# saveall
gv.ranseed(1)
integ = vegas.Integrator([[-1, 1], [0, 1], [0, 1], [0, 1]])

result = integ(f, nitn=5, neval=1000, saveall='saveall.pkl')
# print(result.summary())
# print('result = %s    Q = %.2f' % (result, result.Q))

import pickle
log_stdout('eg9b.out')
with open('saveall.pkl', 'rb') as ifile:
    result, integ = pickle.load(ifile)
new_result = integ(f)

print('\nNew results:')
print(new_result.summary())

print('\nCombined results:')
result.extend(new_result)
print(result.summary())
print('Combined result = %s    Q = %.2f' % (result, result.Q))

import os 
os.remove('save.pkl')
os.remove('saveall.pkl')
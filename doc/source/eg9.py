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
unlog_stdout()

gv.ranseed(124567)
integrator = vegas.Integrator([[0, 1], [0, 1], [0, 1]])
@vegas.rbatchintegrand
def f(x):
   return [x[0],   x[0] * 1e50, x[0] * 1e100]

# result = integrator(f, nitn=3, neval=10000)
result = integrator(f, nitn=15, neval=1000, save='save.pkl')
log_stdout('eg9c.out')
print(result.summary())

print("result (15 itns) =", result)
unlog_stdout()
log_stdout('eg9d.out')
result15 = pickle.load(open('save.pkl', 'rb'))
result8 = vegas.ravg(result15.itn_results[:8])
print("result (8 itns) =", result8)
unlog_stdout()

# not used even though it works well
import lsqfit 
wavg_result15 = lsqfit.wavg(result15.itn_results)
print('lsqfit result (15 itns) =', wavg_result15)

import os 
os.remove('save.pkl')
os.remove('saveall.pkl')
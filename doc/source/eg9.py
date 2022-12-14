import numpy as np
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
gv.ranseed(12578)
log_stdout('eg9a.out')
integ = vegas.Integrator([[-2, 2], [0, 2], [0, 2], [0, 2]])

result = integ(f, nitn=10, neval=1000, save='save.pkl')
print(result.summary())
print('result = %s    Q = %.2f' % (result, result.Q))
unlog_stdout()
# print(integ.settings())
log_stdout('eg9b.out')
import pickle
with open('save.pkl', 'rb') as ifile:
    result = pickle.load(ifile)
result = vegas.ravg(result.itn_results[5:])
print(result.summary())
print('result = %s    Q = %.2f' % (result, result.Q))
unlog_stdout()

# saveall
gv.ranseed(12578)
integ = vegas.Integrator([[-2, 2], [0, 2], [0, 2], [0, 2]])

result = integ(f, nitn=10, neval=1000, saveall='saveall.pkl')
# print(result.summary())
# print('result = %s    Q = %.2f' % (result, result.Q))

import pickle
log_stdout('eg9c.out')
with open('saveall.pkl', 'rb') as ifile:
    result, integ = pickle.load(ifile)
result = vegas.ravg(result.itn_results[5:])
new_result = integ(f, nitn=5)

print('\nNew results:')
print(new_result.summary())

print('\nCombined results:')
result.extend(new_result)
print(result.summary())
print('Combined result = %s    Q = %.2f' % (result, result.Q))
unlog_stdout()



# RMAX = (4 * 0.5**2) ** 0.5

# @vegas.rbatchintegrand
# def f(x):
#     dx2 = 0.0
#     for d in range(4):
#         dx2 += (x[d] - 0.5) ** 2
#     I = np.exp(-dx2**0.5 * 100) * 1013.2118364296088
#     # add I to appropriate bin in dI
#     dI = np.zeros((4, x.shape[-1]), dtype=float)
#     dr = RMAX / (len(dI)+25)
#     j = np.floor(dx2 ** 0.5 / dr)
#     for i in range(len(dI)):
#         dI[i, j==i] = I[j==i]
#     return np.array([I] + list(dI))

# gv.ranseed(12)
# integ = vegas.Integrator(4 * [[0, 1]])
# # result = integ(f, neval=100000, nitn=5, save='save.pkl')
# result = integ(f, neval=10000, nitn=15, save='save.pkl')
# log_stdout('eg9c.out')
# print(result.summary())
# print('result =', result)
# unlog_stdout()
# print(sum(result[1:]) / result[0])

# log_stdout('eg9d.out')
# result8 = vegas.ravg(pickle.load(open('save.pkl', 'rb')).itn_results[9:], weighted=True)
# print(result8.summary())
# print('result =', result8)
# unlog_stdout()
# print(sum(result8[1:]) / result8[0])
import os 
os.remove('save.pkl')
os.remove('saveall.pkl')

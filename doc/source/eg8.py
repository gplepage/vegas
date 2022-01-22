import vegas
import numpy as np
import gvar as gv
from outputsplitter import log_stdout, unlog_stdout

log_stdout('eg8a.out')

integ = vegas.Integrator(2 * [(0., np.pi/2.)])
gv.ranseed(123)
@vegas.rbatchintegrand
def f(x):
    Ia_num =  np.exp(-1e2 * np.sin(x[1])) / ((x[0] - x[1])**2 + 0.01)
    Ib = 1 /  (x[0]**2 + 0.01) / (np.pi/2)
    Ia_den =  np.exp(-1e2 * np.sin(x[1])) / (np.pi/2)
    return dict(Ia_num=Ia_num, Ia_den=Ia_den, Ib=Ib)

w = integ(f, neval=20000, nitn=10)
r = integ(f, neval=20000, nitn=5)
print(r.summary(True))
print('Ia =', r['Ia_num'] / r['Ia_den'], 'Ia - Ib =', r['Ia_num'] / r['Ia_den'] - r['Ib'])

unlog_stdout()
log_stdout('eg8b.out')

integ = vegas.Integrator(2 * [(0., np.pi/2.)])
gv.ranseed(123)
@vegas.rbatchintegrand
def f(x, jac):
    Ia_num =  np.exp(-1e2 * np.sin(x[1])) / ((x[0] - x[1])**2 + 0.01)
    Ia_den =  np.exp(-1e2 * np.sin(x[1])) / jac[0]
    Ib = 1 /  (x[0]**2 + 0.01) / jac[1]
    return dict(Ia_num=Ia_num, Ia_den=Ia_den, Ib=Ib)

w = integ(f, uses_jac=True, neval=20000, nitn=10)
r = integ(f, uses_jac=True, neval=20000, nitn=5)
print(r.summary(True))
print('Ia =', r['Ia_num'] / r['Ia_den'], 'Ia - Ib =', r['Ia_num'] / r['Ia_den'] - r['Ib'])

unlog_stdout()

# integ = vegas.Integrator(2 * [(0., np.pi/2.)])
# gv.ranseed(123)
# @vegas.rbatchintegrand
# def f(x, jac):
#     Ia_num =  np.exp(-1e2 * np.sin(x[1])) / ((x[0] - x[1])**2 + 0.01)
#     Ib = 1 /  (x[0]**2 + 0.01) / jac[1]
#     Ia_den =  np.exp(-1e2 * np.sin(x[1])) / jac[0]
#     return dict(Ia_num=Ia_num, Ia_den=Ia_den, Ib=Ib)

# w = integ(f, uses_jac=True, neval=20000, nitn=10)
# r = integ(f, uses_jac=True, neval=20000, nitn=5)
# print(r.summary(True))
# print('Ia =', r['Ia_num'] / r['Ia_den'], 'Ia - Ib =', r['Ia_num'] / r['Ia_den'] - r['Ib'])

# integ = vegas.Integrator(2 * [(0., np.pi/2.)])
# gv.ranseed(123)
# @vegas.rbatchintegrand
# def f(x):
#     Ia_num =  np.exp(-1e2 * np.sin(x[1])) / ((x[0] - x[1])**2 + 0.01)
#     Ib = 1 /  (x[0]**2 + 0.01) / (np.pi/2)
#     Ia_den =  np.exp(-1e2 * np.sin(x[1])) / (np.pi/2)
#     return dict(Ia_num=Ia_num, Ia_den=Ia_den, Ib=Ib)

# w = integ(f, neval=20000, nitn=10)
# r = integ(f, neval=20000, nitn=5)
# print(r.summary(True))
# print('Ia =', r['Ia_num'] / r['Ia_den'], 'Ia - Ib =', r['Ia_num'] / r['Ia_den'] - r['Ib'])

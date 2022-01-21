import vegas
import numpy as np
import gvar as gv
from outputsplitter import log_stdout, unlog_stdout

log_stdout('eg8a.out')

integ = vegas.Integrator(2 * [(0., 2.)])
gv.ranseed(123)
@vegas.rbatchintegrand
def f(x):
    Ia = 1e3 * np.exp(-1e3 * x[1]) / ((x[0] - x[1])**2 + 0.01)
    Ib = 0.5 /  (x[0]**2 + 0.01) 
    return dict(Ia=Ia, Ib=Ib)

w = integ(f, neval=20000, nitn=10)
r = integ(f, neval=20000, nitn=5)
print(r.summary(True))
print('Ia - Ib =', r['Ia'] - r['Ib'])

unlog_stdout()

log_stdout('eg8b.out')

integ = vegas.Integrator(2 * [(0., 2.)])
gv.ranseed(123)
@vegas.rbatchintegrand
def f(x, jac):
    Ia = 1e3 * np.exp(-1e3 * x[1]) / ((x[0] - x[1])**2 + 0.01)
    Ib = 1 /  (x[0]**2 + 0.01) / jac[1]
    return dict(Ia=Ia, Ib=Ib)

w = integ(f, uses_jac=True, neval=20000, nitn=10)
r = integ(f, uses_jac=True, neval=20000, nitn=5)
print(r.summary(True))
print('Ia - Ib =', r['Ia'] - r['Ib'])
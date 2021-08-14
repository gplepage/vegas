import vegas
import gvar as gv
from outputsplitter import log_stdout, unlog_stdout

gv.ranseed(123456789)
log_stdout('eg7.out')

# multi-dimensional Gaussian distribution
g = gv.BufferDict()
g['a'] = gv.gvar([0., 1.], [[1., 0.99], [0.99, 1.]])
g['b'] = gv.gvar('1(1)')

# integrator for expectation values in distribution g
g_expval = vegas.PDFIntegrator(g)

# want expectation value of [fp, fp**2]
@vegas.rbatchintegrand
def f_f2(p):
    a = p['a']
    b = p['b']
    fp = a[0] * a[1] + b
    return [fp, fp ** 2]

# <f_f2> in distribution g
results = g_expval(f_f2, neval=2000, nitn=5)
print(results.summary())
print('results =', results, '\n')

# mean and standard deviation of f's distribution
fmean = results[0]
fsdev = gv.sqrt(results[1] - results[0] ** 2)
print ('fp.mean =', fmean, '   fp.sdev =', fsdev)
print ("Gaussian approx'n for fp =", f_f2(g)[0], '\n')

# g's pdf norm (should be 1 in this case)
print('PDF norm =', results.pdfnorm)
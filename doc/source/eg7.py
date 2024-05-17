import vegas
import gvar as gv
from outputsplitter import log_stdout, unlog_stdout

gv.ranseed(1235)
log_stdout('eg7.out')

# multi-dimensional Gaussian distribution
g = gv.BufferDict()
g['a'] = gv.gvar([2., 1.], [[1., 0.99], [0.99, 1.]])
g['fb(b)'] = gv.BufferDict.uniform('fb', 0.0, 2.0)

# integrator for expectation values in distribution g
g_expval = vegas.PDFIntegrator(g)

# adapt the integrand to the PDF
g_expval(neval=10_000, nitn=5)

# want expectation value of [fp, fp**2]
@vegas.rbatchintegrand
def f_f2(p):
    a = p['a']
    b = p['b']
    fp = a[0] * a[1] + 3 * b
    return [fp, fp ** 2]

# <f_f2> in distribution g
r = g_expval(f_f2, neval=10_000, nitn=5, adapt=False)
print(r.summary())
print('results =', r, '\n')

# mean and standard deviation of f's distribution
fmean = r[0]
fsdev = gv.sqrt(r[1] - r[0] ** 2)
print ('fp.mean =', fmean, '   fp.sdev =', fsdev)
print ("Gaussian approx'n for fp =", f_f2(g)[0], '\n')

# g's pdf norm (should be 1 in this case)
print('PDF norm =', r.pdfnorm)
unlog_stdout()

log_stdout('eg7a.out')
# integrator for expectation values in distribution g
g_expval = vegas.PDFIntegrator(g)

# adapt the integrand to the PDF
g_expval(neval=10_000, nitn=5)

@vegas.rbatchintegrand
def f(p):
    a = p['a']
    b = p['b']
    fp = a[0] * a[1] + 3 * b
    return dict(a=a, b=b, fp=fp)

# <f> in distribution g and other measures
r = g_expval.stats(f, moments=True, histograms=True)
print('results =', r)
print('   f(g) =', f(g))
print('\ncorrelation matrix:')
print(gv.evalcorr([r['a'][0], r['a'][1], r['b'], r['fp']]))
unlog_stdout()
print()
log_stdout('eg7b.out')
print('Statistics for fp:')
print(r.stats['fp'])
unlog_stdout()
plt = r.stats['fp'].plot_histogram()
plt.xlabel('a[0] * a[1] + 3 * b')
plt.show()

import corner
import gvar as gv 
import numpy as np
import matplotlib.pyplot as plt
import vegas

np.random.seed(123)

@vegas.rbatchintegrand
def g_pdf(p):
    ans = 0
    h = 1.
    for p0 in [0.3, 0.6]:
        ans += h * np.exp(-np.sum((p-p0)**2, axis=0)/2/.01)
        h /= 2
    return ans

g_param = gv.gvar([0.5, 0.5], [[.25, .2], [.2, .25]])
g_ev = vegas.PDFIntegrator(param=g_param, pdf=g_pdf)

# adapt integrator to g_pdf(p) and evaluate <p>
g_ev(neval=4000, nitn=10)
r = g_ev.stats()
print('<p> =', r, '(vegas)')

# sample g_pdf(p) and use sample to evaluate <p>
wgts, p_samples = g_ev.sample(nbatch=40_000)
p_avg = np.sum(wgts[None, :] * p_samples, axis=1)
cosp_avg = np.sum(wgts[None, :] * np.cos(p_samples), axis=1)
print('<p> =', p_avg, '(sample)')
print('<cos(p)> =', cosp_avg, '(sample)')

# cosp_samples = np.cos(p_samples[0])
# all_samples = np.concatenate([p_samples, cosp_samples[None, :]], axis=0)
corner.corner(
    data=p_samples.T, weights=wgts, labels=['p[0]', 'p[1]'],
    range=2 * [0.999], show_titles=True, quantiles=[0.16, 0.5, 0.84],
    )
plt.savefig('bimodal.png', bbox_inches='tight')
plt.show()
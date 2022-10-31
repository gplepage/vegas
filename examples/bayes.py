import matplotlib.pyplot as plt
import numpy as np

import gvar as gv
import lsqfit
import vegas

numpy = np

# options (default is 2nd in each case)
SHOW_PLOT = True 
SHOW_PLOT = False
USE_FIT = True    # integration errors much smaller in this case
USE_FIT = False
W_SHAPE = 19
W_SHAPE = ()

gv.ranseed(1)

def main():
    x = np.array([
        0.2, 0.4, 0.6, 0.8, 1.,
        1.2, 1.4, 1.6, 1.8, 2.,
        2.2, 2.4, 2.6, 2.8, 3.,
        3.2, 3.4, 3.6, 3.8
        ])
    y = gv.gvar([
        '0.38(20)', '2.89(20)', '0.85(20)', '0.59(20)', '2.88(20)',
        '1.44(20)', '0.73(20)', '1.23(20)', '1.68(20)', '1.36(20)',
        '1.51(20)', '1.73(20)', '2.16(20)', '1.85(20)', '2.00(20)',
        '2.11(20)', '2.75(20)', '0.86(20)', '2.73(20)'
        ])
    prior = make_prior(W_SHAPE)

    # modified probability density function
    mod_pdf = ModifiedPDF(data=(x, y), fcn=fitfcn, prior=prior)

    # integrator for expectation values with modified PDF
    # nproc=8 makes things go only a little bit faster
    if USE_FIT:
        fit = lsqfit.nonlinear_fit(data=(x,y), prior=prior, fcn=fitfcn)
        expval = vegas.PDFIntegrator(fit.p, pdf=mod_pdf, nproc=8)
    else:
        expval = vegas.PDFIntegrator(prior, pdf=mod_pdf, nproc=8)

    # adapt integrator to pdf
    nitn = 10
    neval = 10_000
    warmup = expval(neval=neval, nitn=nitn)

    # calculate expectation values
    # N.B. nproc > 1 slows things down (integrand too simple, neval too small)
    # but including it here to make sure it works.
    results = expval(g, neval=neval, nitn=nitn, adapt=False)
    print(results.summary())

    print('vegas results:')
    # c[i]
    cmean = results['c_mean']
    cov = results['c_outer'] - np.outer(cmean, cmean)
    cov = (cov + cov.T) / 2
    # w, b
    wmean, w2mean = results['w']
    wsdev = (w2mean - wmean**2)**0.5
    bmean, b2mean = results['b']
    bsdev = (b2mean - bmean**2)**0.5
    print('  c means =', cmean)
    print('  c cov =', str(cov).replace('\n', '\n' + 10 * ' '))
    print('  w mean =', str(wmean).replace('\n', '\n' + 11 * ' '))
    print('  w sdev =', str(wsdev).replace('\n', '\n' + 11 * ' '))
    print('  b mean =', bmean)
    print('  b sdev =', bsdev)
    # Bayes Factor
    print('\n  logBF =', np.log(results.pdfnorm))
    
    print('\nCombine vegas errors with covariances for final results:')
    # N.B. vegas errors are actually insignificant compared to covariances
    c = cmean + gv.gvar(np.zeros(cmean.shape), gv.mean(cov))
    w = wmean + gv.gvar(np.zeros(np.shape(wmean)), gv.mean(wsdev))
    b = bmean + gv.gvar(np.zeros(np.shape(bmean)), bsdev.mean)
    print('  c =', c)
    print('  corr(c) =', str(gv.evalcorr(c)).replace('\n', '\n' + 12*' '))
    print('  w =', str(w).replace('\n', '\n' + 6*' '))
    print('  b =', b, '\n')

    ### Plot results
    make_plot(data=(x, y), prior=prior, c=c)

# evaluating expectation value of g(p)
@vegas.rbatchintegrand
def g(p):
    try:
        w = p['w']
    except KeyError:
        print(type(p), p)
        exit(0)
    c = p['c']
    c_outer = c[:, None] * c[None,:]
    b = p['b']
    return dict(w=[w, w**2] , c_mean=c, c_outer=c_outer, b=[b, b**2])

@vegas.rbatchintegrand
class ModifiedPDF:
    """ Modified PDF to account for measurement failure. """
    def __init__(self, data, fcn, prior):
        x, y = data
        # add rbatch index to arrays
        self.x = x[:, None]
        self.y = y[:, None]
        self.fcn = fcn
        self.prior = gv.BufferDict()
        self.prior['c'] = prior['c'][:, None]
        # check whether one w or lots of w's
        if np.shape(prior['gw(w)']) != ():
            self.prior['gw(w)'] = prior['gw(w)'][:, None]
        else:
            self.prior['gw(w)'] = prior['gw(w)']
        self.prior['gb(b)'] = prior['gb(b)']

    # support pickling (needed because of GVars in y and prior) 
    # allows nproc>1
    def __getstate__(self):
        return gv.dumps(self.__dict__)

    def __setstate__(self, bstr):
        self.__dict__ = gv.loads(bstr)

    def __call__(self, p):
        y_fx = self.y - self.fcn(self.x, p)
        data_pdf1 = self.gaussian_pdf(y_fx)
        data_pdf2 = self.gaussian_pdf(y_fx, broaden=p['b'])
        prior_pdf = np.prod(self.gaussian_pdf(p['c'] - self.prior['c']), axis=0)
        # Gaussians for gw(w) and gb(b)
        if np.shape(self.prior['gw(w)']) == ():
            prior_pdf *= self.gaussian_pdf(p['gw(w)'] - self.prior['gw(w)'])
        else:
            prior_pdf *= np.prod(self.gaussian_pdf(p['gw(w)'] - self.prior['gw(w)']), axis=0)
        prior_pdf *= self.gaussian_pdf(p['gb(b)'] - self.prior['gb(b)'])
        # p['w'] derived (automatically) from p['gw(w)']
        w = p['w']
        return np.prod((1. - w) * data_pdf1 + w * data_pdf2, axis=0) * prior_pdf

    @staticmethod
    def gaussian_pdf(x, broaden=1.):
        xmean = gv.mean(x)
        xvar = gv.var(x) * broaden ** 2
        return gv.exp(-xmean ** 2 / 2. /xvar) / gv.sqrt(2 * np.pi * xvar)

def fitfcn(x, p):
    c = p['c']
    return c[0] + c[1] * x

def make_prior(w_shape=()):
    prior = gv.BufferDict()
    prior['c'] = gv.gvar(['0(5)', '0(5)'])
    prior['gw(w)'] = gv.BufferDict.uniform('gw', 0., 1., shape=w_shape)
    prior['gb(b)'] = gv.BufferDict.uniform('gb', 5., 20.)
    return prior

def make_plot(data, prior, c):
    if not SHOW_PLOT:
        return
    # plot data
    x, y = data
    plt.errorbar(x, gv.mean(y), gv.sdev(y), fmt='o', c='b')

    # plot lsqfit fit
    fit = lsqfit.nonlinear_fit(data=(x,y), fcn=fitfcn, prior=prior)
    xline = np.linspace(x[0], x[-1], 100)
    yline = fitfcn(xline, fit.pmean)
    plt.plot(xline, gv.mean(yline), 'k:')
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.savefig('outliers1.png', bbox_inches='tight')
    # plt.show()

    # add modified fit to plot
    yline = fitfcn(xline, dict(c=c))
    plt.plot(xline, gv.mean(yline), 'r--')
    yp = gv.mean(yline) + gv.sdev(yline)
    ym = gv.mean(yline) - gv.sdev(yline)
    plt.fill_between(xline, yp, ym, color='r', alpha=0.2)
    plt.savefig('outliers2.png', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()
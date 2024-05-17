import numpy as np
import sys

import gvar as gv
import vegas

try:
    import corner
except ImportError:
    corner = None

try:
    import lsqfit 
except ImportError:
    lsqfit = None

if sys.argv[1:]:
    SHOW_PLOTS = eval(sys.argv[1])   # display picture of grid ?
else:
    SHOW_PLOTS = True


# options (default is 2nd in each case)
USE_FIT = True
USE_FIT = False
FIT_KEYS = ['c', 'w']
FIT_KEYS = ['c', 'b', 'w']
W_SHAPE = 19
W_SHAPE = ()

gv.ranseed(123)

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

    # evaluate expectation value of g(p)[k] for k='w','c'
    @vegas.rbatchintegrand
    def g(p):
        return {k:p[k] for k in FIT_KEYS}

    # integrator for expectation values with modified PDF
    if USE_FIT and lsqfit is not None:
        fit = lsqfit.nonlinear_fit(data=(x,y), prior=prior, fcn=fitfcn)
        expval = vegas.PDFIntegrator(fit.p, pdf=mod_pdf)
    else:
        expval = vegas.PDFIntegrator(prior, pdf=mod_pdf)

    # adapt integrator to pdf
    nitn = 10
    if W_SHAPE == ():
        nstrat = [10,10,2,2]
    else:
        nstrat = [20,20] + W_SHAPE * [1] + [1]
    warmup = expval(nstrat=nstrat, nitn=2 * nitn)

    # calculate expectation values
    results = expval.stats(g, nitn=nitn)
    print(results.summary())
    
    # print out results
    print('Bayesian fit results:')
    for k in results:
        print(f'      {k} = {results[k]}')
        if k == 'c':
            # correlation matrix for c
            print(
                ' corr_c =',
                np.array2string(gv.evalcorr(results['c']), prefix=10 * ' ', precision=3),
                )

    # Bayes Factor
    print('\n  logBF =', np.log(results.pdfnorm))

    # Plot results
    make_plot(x, y, prior, results['c'])
    if W_SHAPE == () and FIT_KEYS == ['c', 'b', 'w'] and USE_FIT == False:
        make_cornerplots(expval, results)

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
        if np.shape(prior['gw(w)']) != ():
            self.prior['gw(w)'] = prior['gw(w)'][:, None]
        else:
            self.prior['gw(w)'] = prior['gw(w)']
        self.prior['gb(b)'] = prior['gb(b)']

    def __call__(self, p):
        if 'b' in FIT_KEYS:
            bwide = p['b']
        else:
            bwide = 10.
        y_fx = self.y - self.fcn(self.x, p)
        data_pdf1 = self.gaussian_pdf(y_fx, broaden=1.)
        data_pdf2 = self.gaussian_pdf(y_fx, broaden=bwide)
        prior_pdf = np.prod(self.gaussian_pdf(p['c'] - self.prior['c']), axis=0)
        if np.shape(self.prior['gw(w)']) == ():
            prior_pdf *= self.gaussian_pdf(p['gw(w)'] - self.prior['gw(w)'])
        else:
            prior_pdf *= np.prod(self.gaussian_pdf(p['gw(w)'] - self.prior['gw(w)']), axis=0)
        if 'b' in FIT_KEYS:
            prior_pdf *= self.gaussian_pdf(p['gb(b)'] - self.prior['gb(b)'])
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

def make_plot(x, y, prior, c):
    if not SHOW_PLOTS:
        return 
    import matplotlib.pyplot as plt
    # plot data
    plt.errorbar(x, gv.mean(y), gv.sdev(y), fmt='o', c='b')

    if lsqfit is not None:
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
    # plt.savefig('outliers2.png', bbox_inches='tight')
    plt.show()

def make_cornerplots(expval, results):
    if not SHOW_PLOTS:
        return 
    import matplotlib.pyplot as plt
    wgts, psamples = expval.sample(nbatch=50_000)
    if corner is not None:
        samples = dict(
            c0=psamples['c'][0], c1=psamples['c'][1],
            b=psamples['b'], w=psamples['w']
            )
        fig = corner.corner(
            data=samples, weights=wgts,  
            range=4*[0.99], show_titles=True, quantiles=[0.16, 0.5, 0.84],
            plot_datapoints=False, fill_contours=True, smooth=1,
            contourf_kwargs=dict(cmap='Blues', colors=None),
            )
        # plt.savefig('outliers3.png', bbox_inches='tight')
        plt.show()
    else:
        csamples = psamples['c']
        c = results['c']
        # range for plot
        s = 2.5
        range = []
        for d in np.arange(2):
            range.append((c[d].mean - s * c[d].sdev, c[d].mean + s * c[d].sdev))
        # 2-d histogram H
        H, xedges, yedges = np.histogram2d(
            csamples[0], csamples[1], bins=20, 
            weights=wgts, density=True, range=range
            )
        # contour plot of histogram H
        xmids = (xedges[1:] + xedges[:-1]) / 2
        ymids = (yedges[1:] + yedges[:-1]) / 2
        X, Y = np.meshgrid(xmids, ymids)
        plt.cla()
        plt.contourf(X, Y, H,)
        plt.xlabel('c[0]')
        plt.ylabel('c[1]')
        # plt.savefig('outliers3.png', bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    main()
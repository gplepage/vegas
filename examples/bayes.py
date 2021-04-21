import vegas 
import gvar as gv 
import numpy as np 

ONE_W = True
SHOW_PLOT = False

if SHOW_PLOT:
    import matplotlib.pyplot as plt 
    import lsqfit
else:
    plt = None

def main():
    # data
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
    data = Data(x, y)
    f = F(data)
    prep_plot(plt, data)
    if ONE_W:
        nstrat = [20, 20, 2]
        nitn_w = 6
        nitn_r = 9
        ranseed = gv.ranseed(12345)
    else:
        nstrat = [100, 100] + len(x) * [1]
        nitn_w = 16 
        nitn_r = 8
        ranseed = gv.ranseed(1)
    print('ranseed = %d' % ranseed)
    itg = vegas.Integrator(
        [(-5, 5), (-5, 5)] + (1 if ONE_W else len(x)) * [(0, 1)], 
        )
    w = itg(f, nstrat=nstrat, nitn=nitn_w)
    nsample = w.sum_neval
    print(w.summary())
    r = itg(f, nstrat=nstrat, nitn=nitn_r)
    nsample += r.sum_neval
    print(r.summary())
    print('neval_tot =', nsample, '   nstrat =', np.array(itg.nstrat))
    print( 'last neval =', itg.last_neval, '   r.sum_neval =', r.sum_neval, '    range =', list(itg.neval_hcube_range))
    print('ninc =', list(itg.map.ninc), '\n')
    p = r['p'] / r['norm']
    covp = r['p*p'] / r['norm'] - np.outer(p, p)
    w = r['w'] / r['norm']
    sigw = np.sqrt(r['w*w'] / r['norm'] - w ** 2)
    print('p =', p, '   w =', w)
    plot_fit(plt, gv.gvar(gv.mean(p), gv.mean(covp)), data, 'b:', color='b', alpha=0.5)
    print('sigp =', np.diagonal(covp) ** 0.5)
    print('corr(p0,p1) =', (covp / np.outer(np.diagonal(covp) ** 0.5, np.diagonal(covp)** 0.5))[0,1])
    print('cov(p,p):\n', covp)
    print('sigw =', sigw)
    print()
    save_plot(plt)

class F(vegas.BatchIntegrand):
    def __init__(self, data, w=None):
        self.data = data 
        self.w = None
    def __call__(self, x):
        p = x[:, :2]
        if self.w is None:
            if ONE_W:
                w = x[:, 2]
            else:
                w = x[:, 2:]
        else:
            w = self.w
        ans = {}
        ans['norm'] = self.data.pdf(p, w)
        ans['p'] = p * ans['norm'][:, None]
        if ONE_W:
            ans['w'] = w * ans['norm']
        else:
            ans['w'] = w * ans['norm'][:, None]
        ans['p*p'] = p[:,None,:] * p[:,:,None] * ans['norm'][:, None, None]
        if ONE_W:
            ans['w*w'] = w * w * ans['norm']
        else:
            ans['w*w'] = w ** 2 * ans['norm'][:, None]
        return ans


def save_plot(plt):
    if  ONE_W and SHOW_PLOT:
        # plt.savefig('bayes.pdf', bbox_inches='tight')
        plt.show()

def plot_fit(plt, p, data,  *args, **kargs):
    if not ONE_W or not SHOW_PLOT:
        return plt
    xline = np.linspace(data.x[0], data.x[-1], 100)
    yline = data.fitfcn(p, x=xline)
    yp = gv.mean(yline) + gv.sdev(yline)
    ym = gv.mean(yline) - gv.sdev(yline)
    if args[0][0] == 'k':
        plt.plot(xline, gv.mean(yline), *args)
    else:
        plt.fill_between(xline,yp,ym, **kargs)

def prep_plot(plt, data):
    if not SHOW_PLOT or not ONE_W:
        return plt
    fit = lsqfit.nonlinear_fit(
        data=gv.gvar(data.y, data.sig), prior=make_prior(), fcn=data.fitfcn
        )
    print(fit)
    if False:
        plt.rc('text',usetex=True)
        plt.rc('font',family='serif', serif=['Times'])
        ebargs = dict(
            fmt='o', mfc='w', alpha=1.0, ms=1.5 * 2.5,
            capsize=1.5 * 1.25, elinewidth=.5 * 1.5, mew=.5 * 1.5
            )
        fw = 3.3   # make bigger
        fh = fw /1.61803399 
        # plt.figure(figsize=(fw,fh))
        plt.rcParams["figure.figsize"] = (fw, fh)
    else:
        plt.rc('font',family='serif')
        ebargs = dict(fmt='o', mfc='w', alpha=1.0)

    plt.errorbar(x=data.x, y=data.y, yerr=data.sig, c='b', **ebargs)
    
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plot_fit(plt, fit.p, data, 'k:') # , color='0.8')
    return plt

class Data:
    def __init__(self, x, y):
        self.x = x 
        self.y = gv.mean(y)
        self.sig = gv.sdev(y)
    def fitfcn(self, p, x=None):
        if x is None:
            x = self.x
        if len(p.shape) > 1:
            return p[:, 0][:, None] + p[:, 1][:, None] * x[None, :]
        else:
            return p[0] + p[1] * x
    def pdf(self, p, w):
        fp = self.fitfcn(p)
        if len(p.shape) > 1:
            y = self.y[None, :]
            if ONE_W:
                w = w[:, None]
            var = (self.sig ** 2)[None, :]
        else:
            y = self.y
            var = self.sig ** 2 
        p1 = (1 - w) * np.exp(-(y - fp) ** 2 / 2 / var) / np.sqrt(2 * np.pi * var)
        p2 = w * np.exp(-(y - fp) ** 2 / 200 / var) / np.sqrt(200 * np.pi * var)
        return np.prod(p1 + p2, axis=-1)

def make_prior():
    prior = gv.gvar(['0(5)', '0(5)'])
    return prior

if __name__ == "__main__":
    main()

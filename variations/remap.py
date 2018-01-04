import numpy as np
import math
import vegas
import lsqfit
import gvar as gv

class AffineFunction(vegas.BatchIntegrand):
    def __init__(self, M, t, region, f):
        self.M = np.asarray(M)
        self.t = np.asarray(t)
        self.region = region
        self.f = f

    def __call__(self, y):
        # y[i] ranges from 0 to 1
        y = np.asarray(y)
        shapey = np.shape(y)
        ny = shapey[-1]

        # transform to +- infty
        ya = np.zeros(shapey, float)
        jac = 1.
        ya = np.log(y / (1. - y))
        jac *= np.prod(1. / (y * (1 - y)), axis=-1)

        # affine transform
        yb = np.tensordot(self.M, ya, [1,-1]).T
        if yb.shape != shapey:
            raise ValueError(yb.shape, shapey)
        if len(shapey) > 1:
            yb += self.t[None,:]
        else:
            yb += self.t
        jac *= abs(np.linalg.det(self.M))

        # go to original coordinates
        x = np.zeros(shapey, float)
        pos = yb >= 0
        neg = yb < 0
        x[pos] = 1. / (np.exp(-yb[pos])     + 1.)
        x[neg] = 1. / (1. / np.exp(yb[neg]) + 1.)
        jac *= np.prod(x * (1.-x), axis=-1)
        for i in range(ny):
            jac *= self.region[i][1] - self.region[i][0]
            if len(shapey) > 1:
                x[:, i] = self.region[i][0] + (self.region[i][1] - self.region[i][0]) * x[:, i]
            else:
                x[i] = self.region[i][0] + (self.region[i][1] - self.region[i][0]) * x[i]
        return self.f(x) * jac

ndim = 4
@vegas.batchintegrand
def fcn(x):
    # main integrand
    ans = 0.0
    scale = 20.
    fac = (scale ** ndim) * 0.10132167575422921535 ** (ndim / 4.)
    for delta in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        dx2 = 0
        for d in range(ndim):
            dx2 += (x[:, d] - delta) ** 2
        ans += np.exp(-dx2 * scale ** 2) * fac
    return ans / 7.


def main():
    neval = 100000
    invdet_fac = 1.
    region = ndim * [(0,1.)]

    # naive MC evaluation of the original fcn
    print vegas.Integrator(region)(fcn, nitn=1, max_nhcube=1, neval=neval)

    # optimize M and t
    def fcn2(z):
        M = np.asarray(z[:-ndim])
        M.shape = ndim,ndim
        t = np.asarray(z[-ndim:])
        aff = AffineFunction(M, t, region, fcn)
        @vegas.batchintegrand
        def aff2(x):
            return aff(x) ** 2
        gv.ranseed(1)
        ans = vegas.Integrator(region)(aff2, max_nhcube=1, nitn=1, neval=neval)
        # print '***', ans
        return ans.mean + invdet_fac / abs(np.linalg.det(M)) # 1000. * (1. - np.linalg.det(M)) ** 2
    z0 = np.array(np.diag(ndim * [1.]).flatten().tolist() + ndim * [0.])
    Mmin = lsqfit.gsl_multiminex(z0, fcn2, tol=1e-1, maxit=1000)
    M, t = Mmin.x[:-ndim], Mmin.x[-ndim:]
    print 'nit', Mmin.nit
    M.shape = (ndim, ndim)
    print 'M',np.linalg.det(M)
    print M
    print 't', t

    # naive MC evaluation in the new space
    aff = AffineFunction(M, t, region, fcn)
    print vegas.Integrator(region)(aff, nitn=1, max_nhcube=1, neval=neval)

    neval /= 10
    # vegas integration of original function
    print vegas.Integrator(region)(fcn, neval=neval, alpha=0.2).summary()

    # vegas integration of transformed function
    print vegas.Integrator(region)(aff, neval=neval, alpha=0.2).summary()


if __name__ == '__main__':
    main()
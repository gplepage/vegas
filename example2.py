import pyximport; pyximport.install()    

from vegas import * 
import numpy as np
import math
import lsqfit
import gvar as gvar

dim = 8
nexp = 3
region = dim * [[0., 1.]]
sig = nexp * [0.1]
ampl = nexp * [1.0]
x0 = np.linspace(0., 1., nexp + 2)[1:-1] 

def slow_python_vec_fcn(x, f, nx):
    # vector loop (over i) explicit
    for i in range(nx):
        ans = 0.0
        for j in range(len(x0)):
            dx2 = 0.0
            for d in range(x.shape[0]):
                dx = x[d, i] - x0[j]
                dx2 += dx * dx
            ans += ampl[j] * numpy.exp(- dx2 / sig[j] ** 2) 
        f[i] = ans # / exact
    return ans

def fast_python_vec_fcn(xx, ff, nx):
    # vector loop implicit
    x = np.asarray(xx)[:, :nx]
    f = np.asarray(ff)
    ans = 0.0
    for j in range(len(x0)):
        dx2 = 0.0
        for d in range(x.shape[0]):
            dx = x[d] - x0[j]
            dx2 += dx * dx
        ans += ampl[j] * np.exp(- dx2 / sig[j] ** 2) 
    f[:nx] = ans #/ self.exact
    return


def main():
    I = Integrator(
        region, 
        neval=50000, 
        nitn=10, 
        beta=0.5, 
        alpha=0.5, 
        analyzer=reporter(0),
        maxinc = 400,
        )
    # tester = VegasTest(I, x0=x0, sig=sig)
    # I_integrate = [                 # select integration options
    #     tester.cython_integrate,
    #     tester.vec_integrate_cython,
    #     tester.integrate_python,
    #     tester.vec_integrate_python,
    #     ][-2]
    # warmup = I_integrate(nitn=10)
    # ans = I_integrate()
    ans = I.vec_integrate(fast_python_vec_fcn)
    print ans

if __name__ == '__main__':
    main()
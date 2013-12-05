# import pyximport; pyximport.install()    

from vegas import * 
import numpy as np
import math
import lsqfit
import gvar as gvar

# def example(x):
# 	return math.exp(-100. * (x[0] ** 2 + x[1] ** 2)) * 100. / 3.14159654

# def vec_example(x, ans, nx):
#     for i in range(nx):
#         ans[i] = math.exp(-100. * (x[0,i] ** 2 + x[1,i] ** 2)) * 100. / 3.14159654

def main():
    dim = 8
    nexp = 3
    region = dim * [[0., 1.]]
    sig = nexp * [0.1]
    x0 = np.linspace(0., 1., nexp + 2)[1:-1] 
    I = Integrator(
        region, 
        neval=1000000, 
        nitn=10, 
        beta=0., 
        alpha=0.5, 
        analyzer=reporter(0),
        # maxinc = 400,
        # nhcube_vec = 30,
        )
    tester = VegasTest(I, x0=x0, sig=sig)
    I_integrate = [                 # select integration options
        tester.cython_integrate,
        tester.vec_integrate_cython,
        tester.integrate_python,
        tester.vec_integrate_python,
        ][1]
    # warmup = I_integrate(nitn=10)
    ans = I_integrate()
    print ans

if __name__ == '__main__':
    if True:
        main()
    else:
        import hotshot, hotshot.stats
        prof = hotshot.Profile("vegas.prof")
        prof.runcall(main)
        prof.close()
        stats = hotshot.stats.load("vegas.prof")
        stats.strip_dirs()
        stats.sort_stats('time', 'calls')
        stats.print_stats(40)

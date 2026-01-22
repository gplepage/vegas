import vegas 
import numpy as np 
from outputsplitter import log_stdout, unlog_stdout
import gvar as gv 

dim = 20
x0list = np.array([
    5 * [0.23] + (dim - 5) * [0.45],
    5 * [0.39] + (dim - 5) * [0.45],
    5 * [0.74] + (dim - 5) * [0.45],
    ])

@vegas.batchintegrand
def f(x):
    ans = 0
    for x0 in x0list:
        dx2 = np.sum((x - x0[None, :]) ** 2, axis=1)
        ans += np.exp(-100 * dx2)
    return ans * 356047712484621.56

def main():
    print(vegas.__version__)
    neval = 2e6
    nitn = 5
    
    gv.ranseed(1)
    integ1 = vegas.Integrator(dim * [[0, 1]])
    integ1(f, neval=neval, nitn=20)       # training
    result1 = integ1(f, nitn=nitn)

    gv.ranseed(12345)
    integ2 = vegas.Integrator(integ1, nstrat=5 * [10] + (dim -  5) * [1], neval=neval)
    result2 = integ2(f, nitn=nitn)
    print(integ2.neval)

    print(70 * '=', 'nstrat=...')
    log_stdout('eg6a.out')
    print(f'DEFAULT:   nstrat = {np.array(integ1.nstrat)} ({np.prod(integ1.nstrat)})')
    print(result1.summary())
    print(f'MODIFIED:   nstrat = {np.array(integ2.nstrat)} ({np.prod(integ2.nstrat)})')
    print(result2.summary())
    unlog_stdout()

    gv.ranseed(1234567)
    integ2 = vegas.restratify(integ1, f, verbose=False)
    result2 = integ2(f, nitn=nitn)

    print(70 * '=', 'restratify(integ1, f)')
    # print(f'DEFAULT:   nstrat = {np.array(integ1.nstrat)} ({np.prod(integ1.nstrat)})')
    # print(result1.summary())
    log_stdout('eg6b.out')
    print(f'MODIFIED:   nstrat = {np.array(integ2.nstrat)} ({np.prod(integ2.nstrat)})')
    print(result2.summary())
    unlog_stdout()

    return
    gv.ranseed(1)
    integ2 = vegas.restratify(integ1, f, below_avg_nstrat=1, verbose=False)
    result2 = integ2(f, nitn=nitn)

    print(70 * '=', 'restratify(integ1, f, below_avg_nstrat=1)')
    # print(f'DEFAULT:   nstrat = {np.array(integ1.nstrat)} ({np.prod(integ1.nstrat)})')
    # print(result1.summary())
    log_stdout('eg6c.out')
    print(f'MODIFIED:   nstrat = {np.array(integ2.nstrat)} ({np.prod(integ2.nstrat)})')
    print(result2.summary())
    unlog_stdout()


if __name__ == "__main__":
    main()
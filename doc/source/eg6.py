import vegas 
import numpy as np 
from outputsplitter import log_stdout, unlog_stdout

dim = 20
r = np.array([
    5 * [0.39] + (dim - 5) * [0.45],
    5 * [0.74] + (dim - 5) * [0.45],
    ])

@vegas.batchintegrand
def f(x):
    ans = 0
    for ri in r:
        dx2 = np.sum((x - ri[None, :]) ** 2, axis=1)
        ans += np.exp(-100 * dx2)
    return ans * 534071568726932.4

def main():
    np.random.seed(12)
    log_stdout('eg6a.out')
    itg = vegas.Integrator(dim * [[0, 1]], alpha=0.25)
    nstrat = 5 * [12] + (dim - 5) * [1]
    itg(f, nitn=15, nstrat=nstrat)
    r = itg(f, nitn=5, nstrat=nstrat)
    print(r.summary())
    print('nstrat =', np.array(itg.nstrat))
    unlog_stdout()
    print()

    log_stdout('eg6b.out')
    itg = vegas.Integrator(dim * [[0, 1]], alpha=0.25)
    neval = 2e6
    itg(f, nitn=15, neval=neval)
    r = itg(f, nitn=5, neval=neval)
    print(r.summary())
    print('nstrat =', np.array(itg.nstrat))
    unlog_stdout()
    print()




# from tools import *


def xmain():
    np.random.seed(123)
    dim = 20
    sdim = 5
    # log_stdout('eg6.out')
    rgn = np.array(dim * [(0., 1.)])
    f = Gaussians([
        # (sdim * [0.23] + (dim - sdim) * [0.5], 10.),
        (sdim * [0.39] + (dim - sdim) * [0.45], 10.),
        (sdim * [0.74] + (dim - sdim) * [0.45], 10.),
        ])
    print(f.g)
    # f = Exponentials([
    #     # (sdim * [0.23] + (dim - sdim) * [0.5], 25.),
    #     (sdim * [0.39] + (dim - sdim) * [0.4], 50.),
    #     (sdim * [0.69] + (dim - sdim) * [0.4], 50.),
    #     ])
    # for kargs in [dict(neval=1e4), dict(nstrat=[13, 13, 13, 1, 1])]:
    # for kargs in [dict(neval=1e5), dict(nstrat=[26, 26, 26, 1, 1])]:
    # for kargs in [dict(neval=1e3), dict(nstrat=[6, 6, 6, 1, 1])]:
    # for kargs in [dict(neval=4e6), dict(nstrat= sdim * [14] + (dim - sdim) * [1])]:
    for kargs in [dict(neval=2e6), dict(nstrat= sdim * [12] + (dim - sdim) * [1])]:
        itg = vegas.Integrator(rgn, alpha=0.25)
        itg(f, nitn=15, **kargs)
        r = itg(f, nitn=5, **kargs)
        print(r.summary())
        print(list(itg.nstrat), list(itg.neval_hcube_range), r.sum_neval)
        print()
    itg.map.show_grid()
    # unlog_stdout()

def xmain():
    dim = 5
    np.random.seed(123)
    itg = vegas.Integrator(dim * [(0, 1)])
    # r = itg(f, nstrat=dim * [4], nitn=5)
    itg(f, nstrat= [6, 6, 6, 1, 1], nitn=4)
    r = itg(f, nstrat= [6, 6, 6, 1, 1], nitn=4)
    # r = itg(f, nstrat= [8, 8, 8, 2, 2], nitn=5)
    # r = itg(f, neval=1e4, nitn=5)
    print(r.summary())
    print(list(itg.nstrat), list(itg.neval_hcube_range), r.sum_neval)

from tools import *

def xmain():
    dim = 3
    s = 100
    g = Gaussians([
        (dim * [0.7],  s / 2**.5 / 3),
        (dim * [0.45], s / 2**.5 / 3),
        ])
    f = Exponentials([
        (dim * [0.7], s),
        (dim * [0.45], s),
        ])
    print(f.g)
    neval = 1e4 
    if True:
        x = g.samples(2000)
        map = vegas.AdaptiveMap(dim * [(0, 1)])
        map.adapt_to_samples(x, f, nitn=10)
        itg = vegas.Integrator(map)
        r = itg(f, neval=neval, nitn=10, alpha=0.1)
        print(r.summary())
        itg.map.show_grid()
    else:
        itg = vegas.Integrator(dim * [(0, 1)])
        w = itg(f, neval=neval, nitn=10, alpha=0.2)
        print(w.summary())
        r = itg(f, neval=neval, nitn=10, alpha=0.2)
        print(r.summary())


if __name__ == "__main__":
    main()
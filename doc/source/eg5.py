import vegas 
import numpy as np 
from outputsplitter import log_stdout, unlog_stdout

@vegas.batchintegrand
def f(x):
    ans = 0
    for c in [0.45, 0.7]:
        dx2 = np.sum((x - c) ** 2, axis=1)
        ans += np.exp(-50 * np.sqrt(dx2))
    return ans * 247366.171

def main():
    dim = 5 
    log_stdout('eg5a.out')
    np.random.seed(123)
    map = vegas.AdaptiveMap(dim * [(0, 1)])
    itg = vegas.Integrator(map, alpha=0.1)
    r = itg(f, neval=1e4, nitn=5)
    print(r.summary())
    unlog_stdout()

    log_stdout('eg5b.out')
    np.random.seed(1234567)
    map = vegas.AdaptiveMap(dim * [(0, 1)])
    x = np.concatenate([
        np.random.normal(loc=0.45, scale=3 / 50, size=(1000, dim)),
        np.random.normal(loc=0.7, scale=3 / 50, size=(1000, dim)),
        ])
    map.adapt_to_samples(x, f, nitn=5)
    itg = vegas.Integrator(map, alpha=0.1)
    r = itg(f, neval=1e4, nitn=5)
    print(r.summary())
    unlog_stdout()

    log_stdout('eg5c.out')
    np.random.seed(123)
    def smc(f, neval, dim):
        " integrates f(y) over dim-dimensional unit hypercube "
        y = np.random.uniform(0,1, (neval, dim))
        fy = f(y)
        return (np.average(fy), np.std(fy) / neval ** 0.5)
    def g(y):
        jac = np.empty(y.shape[0], float)
        x = np.empty(y.shape, float)
        map.map(y, x, jac)
        return jac * f(x)

    # with map
    r = smc(g, 50_000, dim)
    print('   SMC + map:', f'{r[0]:.3f} +- {r[1]:.3f}')

    # without map
    r = smc(f, 50_000, dim)
    print('SMC (no map):', f'{r[0]:.3f} +- {r[1]:.3f}')


def xmain():
    dim = 5  
    s = 50
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
import numpy as np 
import vegas 
import math
from outputsplitter import log_stdout, unlog_stdout

import gvar 
gvar.ranseed(12345)

DIM = 5

@vegas.rbatchintegrand
def f(xd):
    r = xd['r']
    phi = xd['phi']
    # construct Euclidean coordinates, Jacobian
    x = np.zeros((DIM,) + np.shape(r), float)
    x[:] = r
    x[1:] *= np.cumprod(np.sin(phi), axis=0)
    jac = np.prod(x[1:-1], axis=0) * r
    x[:-1] *= np.cos(phi)
    # calculate contribution to sphere's volume
    return jac

def main():
    log_stdout('eg10.out')
    itg = vegas.Integrator(dict(
        r=(0,1), 
        phi=(DIM - 2) * [(0, np.pi)] + [(0, 2 * np.pi)]
        )) #, analyzer=vegas.reporter())
    warmup = itg(f, neval=1000, nitn=10)#, alpha=0.2)
    volume = itg(f, neval=1000, nitn=10)#, alpha=0.2)
    print(itg.settings(), '\n')
    print(f'volume(dim={DIM}) = {volume}')
    unlog_stdout()
    print(volume.summary())
    exact = np.pi ** (DIM/2) / math.gamma(DIM / 2 + 1)
    print(volume / exact)



main()
from __future__ import print_function   # makes this work for python2 and 3

import vegas
import ffcn_f2py
import numpy as np

USE_BATCH = True

def main():
    integ = vegas.Integrator(4 *[[0,1]])
    if USE_BATCH:
        batch_fcn = vegas.batchintegrand(ffcn_f2py.batch_fcn)
        print(integ(batch_fcn, neval=1e4, nitn=10).summary())
        print(integ(batch_fcn, neval=1e4, nitn=10).summary())
    else:
        print(integ(ffcn_f2py.fcn, neval=1e4, nitn=10).summary())
        print(integ(ffcn_f2py.fcn, neval=1e4, nitn=10).summary())

if __name__ == '__main__':
    import numpy as np
    np.random.seed(9)
    main()
import vegas
import numpy as np

from ffcn_cffi import ffi, lib

@vegas.batchintegrand
def batch_f(x):
    n, dim = x.shape
    ans = np.empty(n, float)
    _x = ffi.cast('double*', x.ctypes.data)
    _ans = ffi.cast('double*', ans.ctypes.data)
    lib.batch_fcn(_ans, _x, n, dim)
    return ans

def main():
    integ = vegas.Integrator(4 * [[0., 1.]])
    print(integ(batch_f, neval=1e5, nitn=10).summary())
    print(integ(batch_f, neval=1e5, nitn=10).summary())

if __name__ == '__main__':
    import numpy as np
    np.random.seed(12)
    main()

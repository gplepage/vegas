# cython: language_level=3
from libc.math cimport exp      # use exp() from C library
import numpy as np

cpdef f(double[::1] x):
    cdef double dx2, x0
    cdef Py_ssize_t d, j
    cdef int dim=4
    cdef int N=1000
    cdef double ans =  0
    cdef double norm = (100. / np.pi) ** 2 / N
    for j in range(N):
        x0 = j / (N - 1.)
        dx2 = 0.0
        for d in range(dim):
            dx2 += (x[d] - x0) ** 2
        ans += exp(-100. * dx2)
    return ans * norm

def fbatch(double[:, ::1] x):
    cdef int i
    cdef double[::1] ans = np.zeros(x.shape[0], float)
    for i in range(x.shape[0]):
        ans[i] = f(x[i])
    return ans

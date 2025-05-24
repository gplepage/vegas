# cython: language_level=3
# Created by G. Peter Lepage (Cornell University) in 12/2013.
# Copyright (c) 2013-25 G. Peter Lepage.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version (see <http://www.gnu.org/licenses/>).
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

cdef class LBatchIntegrand:
    cdef public object fcn

cdef class RBatchIntegrand:
    cdef public object fcn

cdef class AdaptiveMap:
    # first index is direction, second is increment
    cdef readonly double[:, ::1] grid
    cdef readonly double[:, ::1] inc
    cdef readonly Py_ssize_t[::1] ninc
    cdef public double[:, ::1] sum_f
    cdef public double[:, ::1] n_f

    cpdef map(self, double[:, ::1] y, double[:, ::1] x, double[::1] J, Py_ssize_t ny=*)
    cpdef invmap(self, double[:, ::1] x, double[:, ::1] y, double[::1] J, Py_ssize_t nx=*)
    cpdef add_training_data(self, double[:, ::1] y, double[::1] f, Py_ssize_t ny=*)

cdef class Integrator:
    # inputs
    cdef public Py_ssize_t neval
    cdef public Py_ssize_t[::1] neval_hcube_range
    cdef public Py_ssize_t min_neval_batch
    cdef public Py_ssize_t maxinc_axis
    cdef public Py_ssize_t max_neval_hcube
    cdef public bint gpu_pad
    cdef public double neval_frac
    cdef public double max_mem
    cdef public Py_ssize_t nitn
    cdef public Py_ssize_t nproc
    cdef public double alpha
    cdef public double rtol
    cdef public double atol
    cdef public bint minimize_mem
    cdef public bint adapt_to_errors
    cdef public double beta
    cdef public bint adapt
    cdef public object analyzer
    cdef public object ran_array_generator
    cdef public bint sync_ran
    cdef public bint mpi
    cdef public bint uniform_nstrat
    cdef public bint uses_jac
    cdef public str save 
    cdef public str saveall
    cdef readonly Py_ssize_t[::1] nstrat
    cdef readonly object xsample
    # generated
    cdef readonly AdaptiveMap map
    cdef readonly object pool
    cdef readonly double sum_sigf
    cdef readonly Py_ssize_t dim
    cdef readonly Py_ssize_t last_neval
    cdef readonly Py_ssize_t min_neval_hcube
    cdef readonly Py_ssize_t nhcube
    # internal work areas
    cdef double[:, ::1] y
    cdef double[:, ::1] x
    cdef double[::1] jac
    cdef double[::1] fdv2
    cdef Py_ssize_t[::1] neval_hcube
    # the following depend upon whether minimize_mem is False or True
    cdef readonly object sigf       # numpy array or h5py Dataset
    cdef readonly object sigf_h5    # None or h5py file


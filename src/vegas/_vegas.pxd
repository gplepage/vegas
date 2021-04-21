# cython: language_level=3
# Created by G. Peter Lepage (Cornell University) in 12/2013.
# Copyright (c) 2013-21 G. Peter Lepage.
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

cimport numpy
# index type for numpy is numpy.npy_intp (signed)
# -- same as numpy.intp_t and Py_ssize_t (usually) in Cython

cdef class BatchIntegrand:
    cdef readonly object fcntype
    cdef public object fcn

# legacy name
cdef class VecIntegrand(BatchIntegrand):
    pass

cdef class AdaptiveMap:
    # first index is direction, second is increment
    cdef readonly double[:, ::1] grid
    cdef readonly double[:, ::1] inc
    cdef readonly numpy.npy_intp[::1] ninc
    cdef double[:, ::1] sum_f
    cdef double[:, ::1] n_f

    cpdef map(self, double[:, ::1] y, double[:, ::1] x, double[::1] J, numpy.npy_intp ny=*)
    cpdef invmap(self, double[:, ::1] x, double[:, ::1] y, double[::1] J, numpy.npy_intp nx=*)
    cpdef add_training_data(self, double[:, ::1] y, double[::1] f, numpy.npy_intp ny=*)

cdef class Integrator:
    # inputs
    cdef public numpy.npy_intp neval
    cdef public numpy.npy_intp[::1] neval_hcube_range
    cdef public numpy.npy_intp nhcube_batch
    cdef public numpy.npy_intp maxinc_axis
    cdef public numpy.npy_intp max_nhcube
    cdef public numpy.npy_intp max_neval_hcube
    cdef public double neval_frac
    cdef public double max_mem
    cdef public numpy.npy_intp nitn
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
    cdef readonly numpy.npy_intp[::1] nstrat
    # generated
    cdef readonly AdaptiveMap map
    cdef readonly double sum_sigf
    cdef readonly numpy.npy_intp dim
    cdef readonly numpy.npy_intp last_neval
    cdef readonly numpy.npy_intp min_neval_hcube
    cdef readonly numpy.npy_intp nhcube
    # internal work areas
    cdef double[:, ::1] y
    cdef double[:, ::1] x
    cdef double[::1] jac
    cdef double[::1] fdv2
    cdef numpy.npy_intp[::1] neval_hcube
    cdef readonly double[::1] sigf


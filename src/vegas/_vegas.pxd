# Created by G. Peter Lepage (Cornell University) in 12/2013.
# Copyright (c) 2013-14 G. Peter Lepage. 
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

ctypedef long INT_TYPE
ctypedef ssize_t INTP_TYPE

cdef class BatchIntegrand:
    cdef readonly object fcntype
    cdef public object fcn

# legacy name
cdef class VecIntegrand(BatchIntegrand):
    pass

cdef class MPIintegrand(BatchIntegrand):
    #cdef public object fcn  from BatchIntegrand
    cdef readonly object comm 
    cdef readonly INTP_TYPE rank 
    cdef readonly INTP_TYPE nproc 
    cdef readonly object seed
    cdef readonly object fcn_shape

cdef class AdaptiveMap:
    # first index is direction, second is increment
    cdef readonly double[:, ::1] grid
    cdef readonly double[:, ::1] inc
    cdef double[:, ::1] sum_f
    cdef double[:, ::1] n_f

    cpdef map(self, double[:, ::1] y, double[:, ::1] x, double[::1] J, INT_TYPE ny=*)
    cpdef add_training_data(self, double[:, ::1] y, double[::1] f, INT_TYPE ny=*)

cdef class Integrator:
    # inputs
    cdef readonly INT_TYPE neval
    cdef readonly INT_TYPE[::1] neval_hcube_range
    cdef readonly INT_TYPE nhcube_batch
    cdef readonly INT_TYPE maxinc_axis
    cdef readonly INT_TYPE max_nhcube
    cdef readonly INT_TYPE max_neval_hcube
    cdef readonly INT_TYPE nitn
    cdef readonly double alpha
    cdef readonly double rtol
    cdef readonly double atol
    cdef readonly bint minimize_mem
    cdef readonly bint adapt_to_errors
    cdef readonly double beta 
    cdef readonly bint adapt
    cdef readonly object analyzer
    cdef readonly object ran_array_generator
    # generated
    cdef readonly AdaptiveMap map 
    cdef readonly double sum_sigf
    cdef readonly INT_TYPE nstrat 
    cdef readonly INT_TYPE min_neval_hcube 
    cdef readonly INT_TYPE dim 
    cdef readonly INT_TYPE last_neval
    cdef readonly INT_TYPE nhcube
    # internal work areas
    cdef double[:, ::1] y
    cdef double[:, ::1] x
    cdef double[::1] jac 
    cdef double[::1] fdv2
    cdef INT_TYPE[::1] neval_hcube
    cdef readonly double[::1] sigf


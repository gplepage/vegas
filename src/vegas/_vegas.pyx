# cython: language_level=3, binding=True
# c#ython: profile=True

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

cimport cython
cimport numpy
from libc.math cimport floor, log, abs, tanh, erf, exp, sqrt, lgamma

import collections
import math
import sys
import time
import warnings

import numpy
import gvar

try:
    import mpi4py
    import mpi4py.MPI
except ImportError:
    mpi4py = None

cdef double TINY = 10 ** (sys.float_info.min_10_exp + 50)  # smallest and biggest
cdef double HUGE = 10 ** (sys.float_info.max_10_exp - 50)  # with extra headroom
cdef double EPSILON = sys.float_info.epsilon * 10.         # roundoff error threshold

# AdaptiveMap is used by Integrator
cdef class AdaptiveMap:
    """ Adaptive map ``y->x(y)`` for multidimensional ``y`` and ``x``.

    An :class:`AdaptiveMap` defines a multidimensional map ``y -> x(y)``
    from the unit hypercube, with ``0 <= y[d] <= 1``, to an arbitrary
    hypercube in ``x`` space. Each direction is mapped independently
    with a Jacobian that is tunable (i.e., "adaptive").

    The map is specified by a grid in ``x``-space that, by definition,
    maps into a uniformly spaced grid in ``y``-space. The nodes of
    the grid are specified by ``grid[d, i]`` where d is the
    direction (``d=0,1...dim-1``) and ``i`` labels the grid point
    (``i=0,1...N``). The mapping for a specific point ``y`` into
    ``x`` space is::

        y[d] -> x[d] = grid[d, i(y[d])] + inc[d, i(y[d])] * delta(y[d])

    where ``i(y)=floor(y*N``), ``delta(y)=y*N - i(y)``, and
    ``inc[d, i] = grid[d, i+1] - grid[d, i]``. The Jacobian for this map, ::

        dx[d]/dy[d] = inc[d, i(y[d])] * N,

    is piece-wise constant and proportional to the ``x``-space grid
    spacing. Each increment in the ``x``-space grid maps into an increment of
    size ``1/N`` in the corresponding ``y`` space. So regions in
    ``x`` space where ``inc[d, i]`` is small are stretched out
    in ``y`` space, while larger increments are compressed.

    The ``x`` grid for an :class:`AdaptiveMap` can be specified explicitly
    when the map is created: for example, ::

        m = AdaptiveMap([[0, 0.1, 1], [-1, 0, 1]])

    creates a two-dimensional map where the ``x[0]`` interval ``(0,0.1)``
    and ``(0.1,1)`` map into the ``y[0]`` intervals ``(0,0.5)`` and
    ``(0.5,1)`` respectively, while ``x[1]`` intervals ``(-1,0)``
    and ``(0,1)`` map into ``y[1]`` intervals ``(0,0.5)`` and  ``(0.5,1)``.

    More typically, an uniform map with ``ninc`` increments 
    is first created: for example, ::

        m = AdaptiveMap([[0, 1], [-1, 1]], ninc=1000)

    creates a two-dimensional grid, with 1000 increments in each direction, 
    that spans the volume ``0<=x[0]<=1``, ``-1<=x[1]<=1``. This map is then 
    trained with data ``f[j]`` corresponding to ``ny`` points ``y[j, d]``,
    with ``j=0...ny-1``, (usually) uniformly distributed in |y| space:
    for example, ::

        m.add_training_data(y, f)
        m.adapt(alpha=1.5)

    ``m.adapt(alpha=1.5)`` shrinks grid increments where ``f[j]``
    is large, and expands them where ``f[j]`` is small. Usually 
    one has to iterate over several sets of ``y``\s and ``f``\s
    before the grid has fully adapted.

    The speed with which the grid adapts is determined by parameter ``alpha``.
    Large (positive) values imply rapid adaptation, while small values (much
    less than one) imply slow adaptation. As in any iterative process that  
    involves random numbers, it is  usually a good idea to slow adaptation 
    down in order to avoid instabilities caused by random fluctuations.

    Args:
        grid (list of arrays): Initial ``x`` grid, where ``grid[d][i]``
            is the ``i``-th node in direction ``d``. Different directions
            can have different numbers of nodes.
        ninc (int or array or ``None``): ``ninc[d]`` (or ``ninc``, if it 
            is a number) is the number of increments along direction ``d`` 
            in the new  ``x`` grid. The new grid is designed to give the same
            Jacobian ``dx(y)/dy`` as the original grid. The default value,
            ``ninc=None``, leaves the grid unchanged.
    """
    def __init__(self, grid, ninc=None):
        cdef numpy.npy_intp i, d, dim
        cdef double griddi
        if isinstance(grid, AdaptiveMap):
            self.ninc = numpy.array(grid.ninc)
            self.inc = numpy.array(grid.inc)
            self.grid = numpy.array(grid.grid)
        else:
            dim = len(grid)
            len_g = numpy.array([len(x) for x in grid])
            if min(len_g) < 2:
                raise ValueError('grid[d] must have at least 2 elements, not {}'.format(min(len_g)))
            self.ninc = len_g - 1
            self.inc = numpy.empty((dim, max(len_g)-1), float)
            self.grid = numpy.empty((dim, self.inc.shape[1] +1), float)
            for d in range(dim):
                for i, griddi in enumerate(sorted(grid[d])):
                    self.grid[d, i] = griddi
                for i in range(len_g[d] - 1):
                    self.inc[d, i] = self.grid[d, i + 1] - self.grid[d, i]
        self.clear()
        if ninc is not None and not numpy.all(ninc == self.ninc):
            if numpy.all(numpy.asarray(self.ninc) == 1):
                self.make_uniform(ninc=ninc)
            else:
                self.adapt(ninc=ninc)

    property  dim:
        " Number of dimensions."
        def __get__(self):
            return self.grid.shape[0]

    def region(self, numpy.npy_intp d=-1):
        """ x-space region.

        ``region(d)`` returns a tuple ``(xl,xu)`` specifying the ``x``-space
        interval covered by the map in direction ``d``. A list containing
        the intervals for each direction is returned if ``d`` is omitted.
        """
        if d < 0:
            return [self.region(d) for d in range(self.dim)]
        else:
            return (self.grid[d, 0], self.grid[d, self.ninc[d]])

    def extract_grid(self):
        " Return a list of lists specifying the map's grid. "
        cdef numpy.npy_intp d 
        grid = []
        for d in range(self.dim):
            ng = self.ninc[d] + 1
            grid.append(list(self.grid[d, :ng]))
        return grid

    def __reduce__(self):
        """ Capture state for pickling. """
        return (AdaptiveMap, (self.extract_grid(),))

    def settings(self, ngrid=5):
        """ Create string with information about grid nodes.

        Creates a string containing the locations of the nodes
        in the map grid for each direction. Parameter
        ``ngrid`` specifies the maximum number of nodes to print
        (spread evenly over the grid).
        """
        cdef numpy.npy_intp d
        ans = []
        if ngrid > 0:
            for d in range(self.dim):
                grid_d = numpy.array(self.grid[d, :self.ninc[d] + 1])
                nskip = int(self.ninc[d] // ngrid)
                if nskip<1:
                    nskip = 1
                start = nskip // 2
                ans += [
                    "    grid[%2d] = %s"
                    % (
                        d,
                        numpy.array2string(
                            grid_d[start::nskip], precision=3,
                            prefix='    grid[xx] = ')
                          )
                    ]
        return '\n'.join(ans) + '\n'

    def random(self, n=None):
        " Create ``n`` random points in |x| space. "
        if n is None:
            y = numpy.random.random(self.dim)
        else:
            y = numpy.random.random((n, self.dim))
        return self(y)

    def make_uniform(self, ninc=None):
        """ Replace the grid with a uniform grid.

        The new grid has ``ninc[d]``  (or ``ninc``, if it is a number) 
        increments along each direction if ``ninc`` is specified.
        If ``ninc=None`` (default), the new grid has the same number 
        of increments in each direction as the old grid.
        """
        cdef numpy.npy_intp i, d
        cdef numpy.npy_intp dim = self.grid.shape[0]
        cdef double[:] tmp
        cdef double[:, ::1] new_grid
        if ninc is None:
            ninc = numpy.asarray(self.ninc)
        elif numpy.shape(ninc) == ():
            ninc = numpy.full(self.dim, int(ninc), dtype=numpy.intp)
        elif numpy.shape(ninc) == (self.dim,):
            ninc = numpy.asarray(ninc)
        else:
            raise ValueError('ninc has wrong shape -- {}'.format(numpy.shape(ninc)))
        if min(ninc) < 1:
            raise ValueError(
                "no of increments < 1 in AdaptiveMap -- %s"
                % str(ninc)
                )
        new_inc = numpy.empty((dim, max(ninc)), numpy.float_)
        new_grid = numpy.empty((dim, new_inc.shape[1] + 1), numpy.float_)
        for d in range(dim):
            tmp = numpy.linspace(self.grid[d, 0], self.grid[d, self.ninc[d]], ninc[d] + 1)
            for i in range(ninc[d] + 1):
                new_grid[d, i] = tmp[i]
            for i in range(ninc[d]):
                new_inc[d, i] = new_grid[d, i + 1] - new_grid[d, i]
        self.ninc = ninc
        self.grid = new_grid 
        self.inc = new_inc 
        self.clear()

    def __call__(self, y=None):
        """ Return ``x`` values corresponding to ``y``.

        ``y`` can be a single ``dim``-dimensional point, or it
        can be an array ``y[i,j, ..., d]`` of such points (``d=0..dim-1``).

        If ``y=None`` (default), ``y`` is set equal to a (uniform) random point
        in the volume.
        """
        if y is None:
            y = numpy.random.uniform(size=self.dim)
        else:
            y = numpy.asarray(y, numpy.float_)
        y_shape = y.shape
        y.shape = -1, y.shape[-1]
        x = 0 * y
        jac = numpy.empty(y.shape[0], numpy.float_)
        self.map(y, x, jac)
        x.shape = y_shape
        return x

    def jac(self, y):
        """ Return the map's Jacobian at ``y``.

        ``y`` can be a single ``dim``-dimensional point, or it
        can be an array ``y[d,i,j,...]`` of such points (``d=0..dim-1``).
        """
        y = numpy.asarray(y)
        y_shape = y.shape
        y.shape = -1, y.shape[-1]
        x = 0 * y
        jac = numpy.empty(y.shape[0], numpy.float_)
        self.map(y, x, jac)
        return jac

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    cpdef map(
        self,
        double[:, ::1] y,
        double[:, ::1] x,
        double[::1] jac,
        numpy.npy_intp ny=-1
        ):
        """ Map y to x, where jac is the Jacobian  (``dx/dy``).

        ``y[j, d]`` is an array of ``ny`` ``y``-values for direction ``d``.
        ``x[j, d]`` is filled with the corresponding ``x`` values,
        and ``jac[j]`` is filled with the corresponding Jacobian
        values. ``x`` and ``jac`` must be preallocated: for example, ::

            x = numpy.empty(y.shape, float)
            jac = numpy.empty(y.shape[0], float)

        Args:
            y (array): ``y`` values to be mapped. ``y`` is a contiguous
                2-d array, where ``y[j, d]`` contains values for points
                along direction ``d``.
            x (array): Container for ``x[j, d]`` values corresponding
                to ``y[j, d]``. Must be a contiguous 2-d array.
            jac (array): Container for Jacobian values ``jac[j]`` (``= dx/dy``)
                corresponding to ``y[j, d]``. Must be a contiguous 1-d array.
            ny (int): Number of ``y`` points: ``y[j, d]`` for ``d=0...dim-1``
                and ``j=0...ny-1``. ``ny`` is set to ``y.shape[0]`` if it is
                omitted (or negative).
        """
        cdef numpy.npy_intp ninc 
        cdef numpy.npy_intp dim = self.inc.shape[0]
        cdef numpy.npy_intp i, iy, d
        cdef double y_ninc, dy_ninc, tmp_jac
        if ny < 0:
            ny = y.shape[0]
        elif ny > y.shape[0]:
            raise ValueError('ny > y.shape[0]: %d > %d' % (ny, y.shape[0]))
        for i in range(ny):
            jac[i] = 1.
            for d in range(dim):
                ninc = self.ninc[d]
                y_ninc = y[i, d] * ninc
                iy = <int>floor(y_ninc)
                dy_ninc = y_ninc  -  iy
                if iy < ninc:
                    x[i, d] = self.grid[d, iy] + self.inc[d, iy] * dy_ninc
                    jac[i] *= self.inc[d, iy] * ninc
                else:
                    x[i, d] = self.grid[d, ninc]
                    jac[i] *= self.inc[d, ninc - 1] * ninc
        return

    cpdef invmap(
        self,
        double[:, ::1] x,
        double[:, ::1] y,
        double[::1] jac,
        numpy.npy_intp nx=-1
        ):
        """ Map x to y, where jac is the Jacobian (``dx/dy``).

        ``y[j, d]`` is an array of ``ny`` ``y``-values for direction ``d``.
        ``x[j, d]`` is filled with the corresponding ``x`` values,
        and ``jac[j]`` is filled with the corresponding Jacobian
        values. ``x`` and ``jac`` must be preallocated: for example, ::

            x = numpy.empty(y.shape, float)
            jac = numpy.empty(y.shape[0], float)

        Args:
            x (array): ``x`` values to be mapped to ``y``-space. ``x`` 
                is a contiguous 2-d array, where ``x[j, d]`` contains 
                values for points along direction ``d``.
            y (array): Container for ``y[j, d]`` values corresponding
                to ``x[j, d]``. Must be a contiguous 2-d array
            jac (array): Container for Jacobian values ``jac[j]`` (``= dx/dy``)
                corresponding to ``y[j, d]``. Must be a contiguous 1-d array
            nx (int): Number of ``x`` points: ``x[j, d]`` for ``d=0...dim-1``
                and ``j=0...nx-1``. ``nx`` is set to ``x.shape[0]`` if it is
                omitted (or negative).
        """
        cdef numpy.npy_intp ninc 
        cdef numpy.npy_intp dim = self.inc.shape[0]
        cdef numpy.npy_intp[:] iy
        cdef numpy.npy_intp i, iyi, d
        cdef double y_ninc, dy_ninc, tmp_jac
        if nx < 0:
            nx = x.shape[0]
        elif nx > x.shape[0]:
            raise ValueError('nx > x.shape[0]: %d > %d' % (nx, x.shape[0]))
        for i in range(nx):
            jac[i] = 1. 
        for d in range(dim):
            ninc = self.ninc[d]
            iy = numpy.searchsorted(self.grid[d, :], x[:, d], side='right')
            for i in range(nx):
                if iy[i] > 0 and iy[i] <= ninc:
                    iyi = iy[i] - 1
                    y[i, d] = (iyi + (x[i, d] - self.grid[d, iyi]) / self.inc[d, iyi]) / ninc
                    jac[i] *= self.inc[d, iyi] * ninc
                elif iy[i] <= 0:
                    y[i, d] = 0. 
                    jac[i] *= self.inc[d, 0] * ninc 
                elif iy[i] > ninc:
                    y[i, d] = 1.0 
                    jac[i] *= self.inc[d, ninc - 1] * ninc 
        return               


    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    cpdef add_training_data(
        self,
        double[:, ::1] y,
        double[::1] f,
        numpy.npy_intp ny=-1,
        ):
        """ Add training data ``f`` for ``y``-space points ``y``.

        Accumulates training data for later use by ``self.adapt()``.
        Grid increments will be made smaller in regions where
        ``f`` is larger than average, and larger where ``f``
        is smaller than average. The grid is unchanged (converged?)
        when ``f`` is constant across the grid.

        Args:
            y (array): ``y`` values corresponding to the training data.
                ``y`` is a contiguous 2-d array, where ``y[j, d]``
                is for points along direction ``d``.
            f (array): Training function values. ``f[j]`` corresponds to
                point ``y[j, d]`` in ``y``-space.
            ny (int): Number of ``y`` points: ``y[j, d]`` for ``d=0...dim-1``
                and ``j=0...ny-1``. ``ny`` is set to ``y.shape[0]`` if it is
                omitted (or negative).
        """
        cdef numpy.npy_intp ninc 
        cdef numpy.npy_intp dim = self.inc.shape[0]
        cdef numpy.npy_intp iy
        cdef numpy.npy_intp i, d
        if self.sum_f is None:
            shape = (self.inc.shape[0], self.inc.shape[1])
            self.sum_f = numpy.zeros(shape, numpy.float_)
            self.n_f = numpy.zeros(shape, numpy.float_) + TINY
        if ny < 0:
            ny = y.shape[0]
        elif ny > y.shape[0]:
            raise ValueError('ny > y.shape[0]: %d > %d' % (ny, y.shape[0]))
        for d in range(dim):
            ninc = self.ninc[d]
            for i in range(ny):
                if y[i, d] > 0 and y[i, d] < 1:
                    iy = <int> floor(y[i, d] * ninc)
                    self.sum_f[d, iy] += abs(f[i])
                    self.n_f[d, iy] += 1
        return

    # @cython.boundscheck(False)
    def adapt(self, double alpha=0.0, ninc=None):
        """ Adapt grid to accumulated training data.

        ``self.adapt(...)`` projects the training data onto
        each axis independently and maps it into ``x`` space.
        It shrinks ``x``-grid increments in regions where the
        projected training data is large, and grows increments
        where the projected data is small. The grid along
        any direction is unchanged if the training data
        is constant along that direction.

        The number of increments along a direction can be
        changed by setting parameter ``ninc`` (array or number).

        The grid does not change if no training data has
        been accumulated, unless ``ninc`` is specified, in
        which case the number of increments is adjusted
        while preserving the relative density of increments
        at different values of ``x``.

        Args:
            alpha (double): Determines the speed with which the grid
                adapts to training data. Large (postive) values imply
                rapid evolution; small values (much less than one) imply
                slow evolution. Typical values are of order one. Choosing
                ``alpha<0`` causes adaptation to the unmodified training
                data (usually not a good idea).
            ninc (int or array or None): The number of increments in the new 
                grid is ``ninc[d]`` (or ``ninc``, if it is a number)
                in direction ``d``. The number is unchanged from the 
                old grid if ``ninc`` is omitted (or equals ``None``, 
                which is the default).
        """
        cdef double[:, ::1] new_grid
        cdef double[::1] avg_f, tmp_f
        cdef double sum_f, acc_f, f_ninc
        cdef numpy.npy_intp old_ninc
        cdef numpy.npy_intp dim = self.grid.shape[0]
        cdef numpy.npy_intp i, j
        cdef numpy.npy_intp[:] new_ninc

        # initialization
        if ninc is None:
            new_ninc = numpy.array(self.ninc)
        elif numpy.shape(ninc) == ():
            new_ninc = numpy.full(dim, int(ninc), numpy.intp)
        elif len(ninc) == dim:
            new_ninc = numpy.array(ninc, numpy.intp)
        else:
            raise ValueError('badly formed ninc = ' + str(ninc))
        if min(new_ninc) < 1:
            raise ValueError('ninc < 1: ' + str(list(new_ninc)))
        if max(new_ninc) == 1:
            new_grid = numpy.empty((dim, 2), numpy.float_)
            for d in range(dim):
                new_grid[d, 0] = self.grid[d, 0]
                new_grid[d, 1] = self.grid[d, self.ninc[d]]
            self.grid = numpy.asarray(new_grid)
            self.inc = numpy.empty((dim, 1), numpy.float_)
            self.ninc = numpy.array(dim * [1])
            for d in range(dim):
                self.inc[d, 0] = self.grid[d, 1] - self.grid[d, 0]
            self.clear()
            return

        # smoothing
        new_grid = numpy.empty((dim, max(new_ninc) + 1), numpy.float_)
        avg_f = numpy.ones(self.inc.shape[1], numpy.float_) # default = uniform
        if alpha > 0 and max(self.ninc) > 1:
            tmp_f = numpy.empty(self.inc.shape[1], numpy.float_)
        for d in range(dim):
            old_ninc = self.ninc[d]
            if self.sum_f is not None and alpha != 0:
                for i in range(old_ninc):
                    if self.n_f[d, i] > 0:
                        avg_f[i] = self.sum_f[d, i] / self.n_f[d, i]
                    else:
                        avg_f[i] = 0.
            if alpha > 0 and old_ninc > 1:
                tmp_f[0] = abs(7. * avg_f[0] + avg_f[1]) / 8.
                tmp_f[old_ninc - 1] = abs(7. * avg_f[old_ninc - 1] + avg_f[old_ninc - 2]) / 8.
                sum_f = tmp_f[0] + tmp_f[old_ninc - 1]
                for i in range(1, old_ninc - 1):
                    tmp_f[i] = abs(6. * avg_f[i] + avg_f[i-1] + avg_f[i+1]) / 8.
                    sum_f += tmp_f[i]
                if sum_f > 0:
                    for i in range(old_ninc):
                        avg_f[i] = tmp_f[i] / sum_f + TINY
                else:
                    for i in range(old_ninc):
                        avg_f[i] = TINY
                for i in range(old_ninc):
                    if avg_f[i] > 0 and avg_f[i] <= 0.99999999:
                        avg_f[i] = (-(1 - avg_f[i]) / log(avg_f[i])) ** alpha

            # regrid
            new_grid[d, 0] = self.grid[d, 0]
            new_grid[d, new_ninc[d]] = self.grid[d, old_ninc]
            i = 0        # new_x index
            j = -1         # self_x index
            acc_f = 0   # sum(avg_f) accumulated
            f_ninc = 0.
            for i in range(old_ninc):
                f_ninc += avg_f[i]
            f_ninc /= new_ninc[d]     # amount of acc_f per new increment
            for i in range(1, new_ninc[d]):
                while acc_f < f_ninc:
                    j += 1
                    if j < old_ninc:
                        acc_f += avg_f[j]
                    else:
                        break
                else:
                    acc_f -= f_ninc
                    new_grid[d, i] = (
                        self.grid[d, j+1]
                        - (acc_f / avg_f[j]) * self.inc[d, j]
                        )
                    continue
                break
        self.grid = numpy.asarray(new_grid)
        self.inc = numpy.empty((dim, self.grid.shape[1] - 1), float)
        for d in range(dim):
            for i in range(new_ninc[d]):
                self.inc[d, i] = self.grid[d, i + 1] - self.grid[d, i]
        self.ninc = numpy.asarray(new_ninc)
        self.clear()

    def clear(self):
        " Clear information accumulated by :meth:`AdaptiveMap.add_training_data`. "
        self.sum_f = None 
        self.n_f = None

    def show_grid(self, ngrid=40, axes=None, shrink=False, plotter=None):
        """ Display plots showing the current grid.

        Args:
            ngrid (int): The number of grid nodes in each
                direction to include in the plot. The default is 40.
            axes: List of pairs of directions to use in
                different views of the grid. Using ``None`` in
                place of a direction plots the grid for only one
                direction. Omitting ``axes`` causes a default
                set of pairings to be used.
            shrink: Display entire range of each axis
                if ``False``; otherwise shrink range to include
                just the nodes being displayed. The default is
                ``False``.
            plotter: :mod:`matplotlib` plotter to use for plots; plots
                are not displayed if set. Ignored if ``None``, and 
                plots are displayed using ``matplotlib.pyplot``.
        """
        if plotter is not None:
            plt = plotter
        else:
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                warnings.warn('matplotlib not installed; cannot show_grid')
                return
        dim = self.dim
        if axes is None:
            axes = []
            if dim == 1:
                axes = [(0, None)]
            for d in range(dim):
                axes.append((d, (d + 1) % dim))
        else:
            if len(axes) <= 0:
                return
            for dx,dy in axes:
                if dx is not None and (dx < 0 or dx >= dim):
                    raise ValueError('bad directions: %s' % str((dx, dy)))
                if dy is not None and (dy < 0 or dy >= dim):
                    raise ValueError('bad directions: %s' % str((dx, dy)))
        fig = plt.figure()
        def plotdata(idx, grid=numpy.asarray(self.grid)):
            dx, dy = axes[idx[0]]
            if dx is not None:
                nskip = int(self.ninc[dx] // ngrid)
                if nskip < 1:
                    nskip = 1
                start = nskip // 2
                xrange = [self.grid[dx, 0], self.grid[dx, self.ninc[dx]]]
                xgrid = grid[dx, start::nskip]
                xlabel = 'x[%d]' % dx
            else:
                xrange = [0., 1.]
                xgrid = None
                xlabel = ''
            if dy is not None:
                nskip = int(self.ninc[dy] // ngrid)
                if nskip < 1:
                    nskip = 1
                start = nskip // 2
                yrange = [self.grid[dy, 0], self.grid[dy, self.ninc[dy]]]
                ygrid = grid[dy, start::nskip]
                ylabel = 'x[%d]' % dy
            else:
                yrange = [0., 1.]
                ygrid = None
                ylabel = ''
            if shrink:
                if xgrid is not None:
                    xrange = [min(xgrid), max(xgrid)]
                if ygrid is not None:
                    yrange = [min(ygrid), max(ygrid)]
            if None not in [dx, dy]:
                fig_caption = 'axes %d, %d' % (dx, dy)
            elif dx is None and dy is not None:
                fig_caption = 'axis %d' % dy
            elif dx is not None and dy is None:
                fig_caption = 'axis %d' % dx
            else:
                return
            fig.clear()
            plt.title(
                "%s   (press 'n', 'p', 'q' or a digit)"
                % fig_caption
                )
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            if xgrid is not None:
                for i in range(len(xgrid)):
                    plt.plot([xgrid[i], xgrid[i]], yrange, 'k-')
            if ygrid is not None:
                for i in range(len(ygrid)):
                    plt.plot(xrange, [ygrid[i], ygrid[i]], 'k-')
            plt.xlim(*xrange)
            plt.ylim(*yrange)

            plt.draw()

        idx = [0]
        def onpress(event, idx=idx):
            try:    # digit?
                idx[0] = int(event.key)
            except ValueError:
                if event.key == 'n':
                    idx[0] += 1
                    if idx[0] >= len(axes):
                        idx[0] = len(axes) - 1
                elif event.key == 'p':
                    idx[0] -= 1
                    if idx[0] < 0:
                        idx[0] = 0
                elif event.key == 'q':
                    plt.close()
                    return
                else:
                    return
            plotdata(idx)

        fig.canvas.mpl_connect('key_press_event', onpress)
        plotdata(idx)
        if plotter is None:
            plt.show()
        else:
            return plt

    def adapt_to_samples(self, x, f, nitn=5, alpha=1.0): # , ninc=None):
        """ Adapt map to data ``{x, f(x)}``.

        Replace grid with one that is optimized for integrating 
        function ``f(x)``. New grid is found iteratively

        Args:
            x (array): ``x[:, d]`` are the components of the sample points 
                in direction ``d=0,1...self.dim-1``.
            f (callable or array): Function ``f(x)`` to be adapted to. If 
                ``f`` is an array, it is assumes to contain values ``f[i]``
                corresponding to the function evaluated at points ``x[i]``.
            nitn (int): Number of iterations to use in adaptation. Default
                is ``nitn=5``.
            alpha (float): Damping parameter for adaptation. Default 
                is ``alpha=1.0``. Smaller values slow the iterative 
                adaptation, to improve stability of convergence.
        """
        cdef numpy.npy_intp i, tmp_ninc, old_ninc
        x = numpy.ascontiguousarray(x)
        if len(x.shape) != 2 or x.shape[1] != self.dim:
            raise ValueError('incompatible shape of x: {}'.format(x.shape))
        if callable(f):
            fx = numpy.ascontiguousarray(f(x))
        else:
            fx = numpy.ascontiguousarray(f)
        if fx.shape[0] != x.shape[0]:
            raise ValueError('shape of x and f(x) mismatch: {} vs {}'.format(x.shape, fx.shape))
        old_ninc = max(max(self.ninc), Integrator.defaults['maxinc_axis'])
        tmp_ninc = min(old_ninc, x.shape[0] / 10.) 
        if tmp_ninc < 2:
            raise ValueError('not enough samples: {}'.format(x.shape[0]))
        y = numpy.empty(x.shape, float)
        jac = numpy.empty(x.shape[0], float)
        for i in range(nitn):
            self.invmap(x, y, jac)
            self.add_training_data(y, (jac * fx) ** 2)
            self.adapt(alpha=alpha, ninc=tmp_ninc)
        if numpy.any(tmp_ninc != old_ninc):
            self.adapt(ninc=old_ninc)



cdef class Integrator(object):
    """ Adaptive multidimensional Monte Carlo integration.

    :class:`vegas.Integrator` objects make Monte Carlo
    estimates of multidimensional functions ``f(x)``
    where ``x[d]`` is a point in the integration volume::

        integ = vegas.Integrator(integration_region)

        result = integ(f, nitn=10, neval=10000)

    The integator makes ``nitn`` estimates of the integral,  each
    using at most ``neval`` samples of the integrand, as it adapts to
    the specific features of the integrand. Successive estimates (iterations)
    typically improve in accuracy until the integrator has fully
    adapted. The integrator returns the weighted average of all
    ``nitn`` estimates, together with an estimate of the statistical
    (Monte Carlo) uncertainty in that estimate of the integral. The
    result is an object of type :class:`RAvg` (which is derived
    from :class:`gvar.GVar`).

    Integrands ``f(x)`` return numbers, arrays of numbers (any shape), or
    dictionaries whose values are numbers or arrays (any shape). Each number
    returned by an integrand corresponds to a different integrand. When
    arrays are returned, |vegas| adapts to the first number
    in the flattened array. When dictionaries are returned,
    |vegas| adapts to the first number in the value corresponding to
    the first key.

    |vegas| can generate integration points in batches for integrands
    built from classes derived from :class:`vegas.BatchIntegrand`, or
    integrand functions decorated by :func:`vegas.batchintegrand`. Batch
    integrands are typically much faster, especially if they are coded in
    Cython or C/C++ or Fortran.

    |Integrator|\s have a large number of parameters but the
    only ones that most people will care about are: the
    number ``nitn`` of iterations of the |vegas| algorithm;
    the maximum number ``neval`` of integrand evaluations per
    iteration; and the damping parameter ``alpha``, which is used
    to slow down the adaptive algorithms when they would otherwise
    be unstable (e.g., with very peaky integrands). Setting parameter
    ``analyzer=vegas.reporter()`` is sometimes useful, as well,
    since it causes |vegas| to print (on ``sys.stdout``)
    intermediate results from each iteration, as they are
    produced. This helps when each iteration takes a long time
    to complete (e.g., longer than an hour) because it allows you to
    monitor progress as it is being made (or not).

    Args:
        map (array, :class:`vegas.AdaptiveMap` or :class:`vegas.Integrator`):
            The integration region  as specified by an array ``map[d, i]``
            where ``d`` is the direction and ``i=0,1`` specify the lower
            and upper limits of integration in direction ``d``.

            ``map`` could also be the integration map from
            another |Integrator|, or that |Integrator|
            itself. In this case the grid is copied from the
            existing integrator.
        nitn (positive int): The maximum number of iterations used to
            adapt to the integrand and estimate its value. The
            default value is 10; typical values range from 10
            to 20.
        neval (positive int): Approximate number of integrand evaluations
            in each iteration of the |vegas| algorithm. Increasing
            ``neval`` increases the precision: statistical errors should
            fall at least as fast as ``sqrt(1./neval)`` and often
            fall much faster.  The default value is 1000;
            real problems often require 10--10,000 times more evaluations
            than this. 
        nstrat (int array): ``nstrat[d]`` specifies the number of
            stratifications to use in direction ``d``. By default this 
            parameter is set automatically, based on parameter ``neval``,
            with ``nstrat[d]`` approximately the same for every ``d``. 
            Specifying ``nstrat`` explicitly makes it possible to 
            concentrate stratifications in directions  where they are most 
            needed. If ``nstrat`` is set but ``neval`` is not, 
            ``neval`` is set equal to ``2*prod(nstrat)/(1-neval_frac)``. 
        alpha (float): Damping parameter controlling the remapping
            of the integration variables as |vegas| adapts to the
            integrand. Smaller values slow adaptation, which may be
            desirable for difficult integrands. Small or zero ``alpha``\s
            are also sometimes useful after the grid has adapted,
            to minimize fluctuations away from the optimal grid.
            The default value is 0.5.
        beta (float): Damping parameter controlling the redistribution
            of integrand evaluations across hypercubes in the
            stratified sampling of the integral (over transformed
            variables). Smaller values limit the amount of
            redistribution. The theoretically optimal value is 1;
            setting ``beta=0`` prevents any redistribution of
            evaluations. The default value is 0.75.
        neval_frac (float): Approximate fraction of function evaluations
            used for adaptive stratified sampling. |vegas| 
            distributes ``(1-neval_frac)*neval``  integrand evaluations 
            uniformly over all hypercubes, with at  least 2 evaluations 
            per hypercube. The remaining ``neval_frac*neval`` 
            evaluations are concentrated in hypercubes where the errors 
            are largest. Increasing ``neval_frac`` makes more integrand 
            evaluations available for adaptive stratified 
            sampling, but reduces the number of hypercubes, which limits
            the algorithm's ability to adapt. Ignored when ``beta=0``. 
            Default is ``neval_frac=0.75``. 
        adapt (bool): Setting ``adapt=False`` prevents further
            adaptation by |vegas|. Typically this would be done
            after training the |Integrator| on an integrand, in order
            to stabilize further estimates of the integral. |vegas| uses
            unweighted averages to combine results from different
            iterations when ``adapt=False``. The default setting
            is ``adapt=True``.
        analyzer: An object with methods

                ``analyzer.begin(itn, integrator)``

                ``analyzer.end(itn_result, result)``

            where: ``begin(itn, integrator)`` is called at the start
            of each |vegas| iteration with ``itn`` equal to the
            iteration number and ``integrator`` equal to the
            integrator itself; and ``end(itn_result, result)``
            is called at the end of each iteration with
            ``itn_result`` equal to the result for that
            iteration and ``result`` equal to the cummulative
            result of all iterations so far.
            Setting ``analyzer=vegas.reporter()``, for
            example, causes vegas to print out a running report
            of its results as they are produced. The default
            is ``analyzer=None``.
        nhcube_batch (positive int): The number of hypercubes (in |y| space)
            whose integration points are combined into a single
            batch to be passed to the integrand, together,
            when using |vegas| in batch mode.
            The default value is 1000. Larger values may be
            lead to faster evaluations, but at the cost of
            more memory for internal work arrays.
        maxinc_axis (positive int): The maximum number of increments
            per axis allowed for the |x|-space grid. The default
            value is 1000; there is probably little need to use
            other values.
        max_nhcube (positive int): Maximum number of |y|-space hypercubes
            used for stratified sampling. Setting ``max_nhcube=1``
            turns stratified sampling off, which is probably never
            a good idea. The default setting (1e9) was chosen to
            correspond to the point where internal work arrays
            become comparable in size to the typical amount of RAM
            available to a processor (in a laptop in 2014).
            Internal memory usage is large only when ``beta>0``
            and ``minimize_mem=False``; therefore ``max_nhcube`` is
            ignored if ``beta=0`` or ``minimize_mem=True``.
        max_neval_hcube (positive int): Maximum number of integrand
            evaluations per hypercube in the stratification. The default
            value is 1e6. Larger values might allow for more adaptation
            (when ``beta>0``), but also can result in large internal
            work arrays.
        max_mem (positive float): Maximum number of floats allowed in 
            internal work arrays (approx.). A ``MemoryError`` is 
            raised if the work arrays are too large, in which case
            one might want to reduce ``max_neval_hcube`` or 
            ``max_nhcube`` (or increase ``max_mem`` if there is 
            enough RAM). Default value is 1e9.
        rtol (float): Relative error in the integral estimate
            at which point the integrator can stop. The default
            value is 0.0 which turns off this stopping condition.
            This stopping condition can be quite unreliable
            in early iterations, before |vegas| has converged.
            Use with caution, if at all.
        atol (float): Absolute error in the integral estimate
            at which point the integrator can stop. The default
            value is 0.0 which turns off this stopping condition.
            This stopping condition can be quite unreliable
            in early iterations, before |vegas| has converged.
            Use with caution, if at all.
        ran_array_generator: Function that generates
            :mod:`numpy` arrays of random numbers distributed uniformly
            between 0 and 1. ``ran_array_generator(shape)`` should
            create an array whose dimensions are specified by the
            integer-valued tuple ``shape``. The default generator
            is ``numpy.random.random``.
        sync_ran (bool): If ``True`` (default), the default random
            number generator is synchronized across all processors when
            using MPI. If ``False``, |vegas| does no synchronization
            (but the random numbers should synchronized some other
            way). 
        adapt_to_errors (bool): 
            ``adapt_to_errors=False`` causes |vegas| to remap the 
            integration variables to emphasize regions where ``|f(x)|`` 
            is largest. This is the default mode.

            ``adapt_to_errors=True`` causes |vegas| to remap
            variables to emphasize regions where the Monte Carlo
            error is largest. This might be superior when
            the number of the number of stratifications (``self.nstrat``)
            in the |y| grid is large (> 100). It is typically
            useful only in one or two dimensions.
        uniform_nstrat (bool): If ``True``, requires that the
            ``nstrat[d]`` be equal for all ``d``. If ``False`` (default), 
            the algorithm maximizes the number of stratifications while 
            requiring ``|nstrat[d1] - nstrat[d2]| <= 1``. This parameter
            is ignored if ``nstrat`` is specified explicitly.
        mpi (bool): Setting ``mpi=False`` disables ``mpi`` support in
            ``vegas`` even if ``mpi`` is available; setting ``mpi=True``
            (default) allows use of ``mpi`` provided module :mod:`mpi4py`
            is installed. This flag is ignored when ``mpi`` is not 
            being used (and so has no impact on performance).
        minimize_mem (bool): When ``True``, |vegas| minimizes
            internal workspace at the cost of extra evaluations of
            the integrand. This can increase execution time by
            50--100% but might be desirable when the number of
            evaluations is very large (e.g., ``neval=1e9``). Normally
            |vegas| uses internal work space that grows in
            proportion to ``neval``. If that work space exceeds
            the size of the RAM available to the processor,
            |vegas| runs much more slowly. Setting ``minimize_mem=True``
            greatly reduces the internal storage used by |vegas|; in
            particular memory becomes independent of ``neval``. The default
            setting (``minimize_mem=False``), however, is much superior
            unless memory becomes a problem. (The large memory is needed
            for adaptive stratified sampling, so memory is not
            an issue if ``beta=0``.)
    """

    # Settings accessible via the constructor and Integrator.set
    defaults = dict(
        map=None,
        neval=1000,         # number of evaluations per iteration
        maxinc_axis=1000,   # number of adaptive-map increments per axis
        nhcube_batch=1000,  # number of h-cubes per batch
        max_nhcube=1e9,     # max number of h-cubes
        max_neval_hcube=1e6,# max number of evaluations per h-cube
        neval_frac=0.75,    # fraction of evaluations used for adaptive stratified sampling
        max_mem=1e9,        # memory cutoff (# of floats)
        nitn=10,            # number of iterations
        alpha=0.5,
        beta=0.75,
        adapt=True,
        minimize_mem=False,
        adapt_to_errors=False,
        uniform_nstrat=False,
        rtol=0,
        atol=0,
        analyzer=None,
        ran_array_generator=numpy.random.random,
        sync_ran=True,
        mpi=True,
        )

    def __init__(Integrator self not None, map, **kargs):
        # N.B. All attributes initialized automatically by cython.
        #      This is why self.set() works here.
        self.sigf = numpy.array([], numpy.float_) # reset sigf (dummy)
        self.sum_sigf = HUGE
        self.neval_hcube_range = None
        self.last_neval = 0
        if isinstance(map, Integrator):
            args = {}
            for k in Integrator.defaults:
                if k == 'map':
                    self.map = AdaptiveMap(map.map)
                else:
                    args[k] = getattr(map, k)
        else:
            args = dict(Integrator.defaults)
            if 'map' in args:
                del args['map']
            self.map = AdaptiveMap(map)
            self.nstrat = numpy.full(self.map.dim, 0, dtype=numpy.intp) # dummy (flags action in self.set())
        args.update(kargs)
        if 'nstrat' in kargs and 'neval' not in kargs and 'neval' in args:
            del args['neval']
        if 'neval' in kargs and 'nstrat' not in kargs and 'nstrat' in args:
            del args['nstrat']
        self.set(args)

    def __reduce__(Integrator self not None):
        """ Capture state for pickling. """
        odict = dict()
        for k in Integrator.defaults:
            if k in ['map']:
                continue
            odict[k] = getattr(self, k)
        return (Integrator, (self.map,), odict)

    def __setstate__(Integrator self not None, odict):
        """ Set state for unpickling. """
        self.set(odict)

    def set(Integrator self not None, ka={}, **kargs):
        """ Reset default parameters in integrator.

        Usage is analogous to the constructor
        for |Integrator|: for example, ::

            old_defaults = integ.set(neval=1e6, nitn=20)

        resets the default values for ``neval`` and ``nitn``
        in |Integrator| ``integ``. A dictionary, here
        ``old_defaults``, is returned. It can be used
        to restore the old defaults using, for example::

            integ.set(old_defaults)
        """
        # 1) reset parameters
        if kargs:
            kargs.update(ka)
        else:
            kargs = ka
        old_val = dict()        # records anything that is changed
        nstrat = None
        for k in kargs:
            if k == 'map':
                old_val['map'] = self.map
                self.map = AdaptiveMap(kargs['map'])
            elif k == 'nstrat':
                if kargs['nstrat'] is None:
                    continue
                old_val['nstrat'] = self.nstrat                
                nstrat = numpy.array(kargs['nstrat'], dtype=numpy.intp)
            elif k == 'sigf':
                old_val['sigf'] = self.sigf 
                self.sigf = numpy.fabs(kargs['sigf'])
                self.sum_sigf = numpy.sum(self.sigf)
            elif k in Integrator.defaults:
                # ignore entry if set to None (useful for debugging)
                if kargs[k] is None:
                    continue
                old_val[k] = getattr(self, k)
                setattr(self, k, kargs[k])
            else:
                raise AttributeError('no attribute named "%s"' % str(k))
        
        # 2) sanity checks
        if nstrat is not None:
            if len(nstrat) != self.map.dim:
                raise ValueError('nstrat[d] has wrong length: %d not %d' % (len(nstrat), self.map.dim))
            if numpy.any(nstrat < 1):
                raise ValueError('bad nstrat: ' + str(numpy.asarray(self.nstrat)))
        if self.neval_frac < 0 or self.neval_frac >= 1:
            raise ValueError('neval_frac = {} but require 0 <= neval_frac < 1'.format(self.neval_frac))
        if 'neval' in kargs and self.neval < 2:
            raise ValueError('neval>2 required, not ' + str(self.neval))
        neval_frac = 0 if (self.beta == 0 or self.adapt_to_errors) else self.neval_frac

        self.dim = self.map.dim

        # 3) determine # strata in each direction
        if nstrat is not None:
            # nstrat specified explicitly
            if len(nstrat) != self.dim or min(nstrat) < 1:
                raise ValueError('bad nstrat = %s' % str(numpy.asarray(nstrat)))
            nhcube = numpy.product(nstrat)
            if 'neval' not in old_val:
                old_val['neval'] = self.neval
                self.neval = 2. * nhcube / (1. - neval_frac)
            elif self.neval < 2. * nhcube / (1. - neval_frac):
                raise ValueError('neval too small: {} < {}'.format(self.neval, 2. * nhcube / (1. - neval_frac)))
        elif 'neval' in old_val or 'neval_frac' in old_val:
            # determine stratification from neval,neval_frac if either was specified
            ns = int(((1 - neval_frac) * self.neval / 2.) ** (1. / self.dim)) # stratifications / axis
            if ns < 1:
                ns = 1
            d = int(
                (numpy.log((1 - neval_frac) * self.neval / 2.) - self.dim * numpy.log(ns))
                / numpy.log(1 + 1. / ns)
                )
            if ((ns + 1)**d * ns**(self.dim-d)) > self.max_nhcube and not self.minimize_mem:
                ns = int(self.max_nhcube ** (1. / self.dim)) - 1
                if ns < 1:
                    ns = 1
                d = int(
                    (numpy.log((1 - neval_frac) * self.neval / 2.) - self.dim * numpy.log(ns))
                    / numpy.log(1 + 1. / ns)
                    )
            if self.uniform_nstrat:
                d = 0
            nstrat = numpy.empty(self.dim, numpy.intp)    
            nstrat[:d] = ns + 1        
            nstrat[d:] = ns 
        else:
            # go with existing grid if none of nstrat, neval and neval_frac changed
            nstrat = self.nstrat
        
        if not numpy.all(numpy.equal(self.nstrat, nstrat)):
            # need to recalculate stratification distribution for beta>0
            self.sum_sigf = HUGE
            self.nstrat = nstrat
        
        # 4) reconfigure vegas map, if necessary
        if self.adapt_to_errors:
            self.map.adapt(ninc=numpy.asarray(self.nstrat))
        else:
            ni = min(int(self.neval / 10.), self.maxinc_axis)   # increments/axis
            ninc = numpy.empty(self.dim, numpy.intp)
            for d in range(self.dim):
                if ni >= self.nstrat[d]:
                    ninc[d] = int(ni / self.nstrat[d]) * self.nstrat[d]
                elif self.nstrat[d] <= self.maxinc_axis:
                    ninc[d] = self.nstrat[d]
                else:
                    self.nstrat[d] = int(self.nstrat[d] / ni) * ni
                    ninc[d] = ni
            if not numpy.all(numpy.equal(self.map.ninc, ninc)):
                self.map.adapt(ninc=ninc)

        # 5) set min_neval_hcube 
        # chosen so that actual neval is close to but not larger than self.neval
        # (unless self.minimize_mem is True in which case it could be larger)
        self.nhcube = numpy.product(self.nstrat) 
        avg_neval_hcube = int(self.neval / self.nhcube)
        if self.nhcube == 1:
            self.min_neval_hcube = int(self.neval / self.nhcube)
        else:
            self.min_neval_hcube = int((1 - neval_frac) * self.neval / self.nhcube)
        if self.min_neval_hcube < 2:
            self.min_neval_hcube = 2

        # 6) allocate work arrays -- these are stored in the
        # the Integrator so that the storage is held between
        # iterations, thereby minimizing the amount of allocating
        # that goes on

        neval_batch = self.nhcube_batch * avg_neval_hcube
        nsigf = (
            self.nhcube_batch if self.minimize_mem else
            self.nhcube
            )
        if self.beta > 0 and self.nhcube > 1 and not self.adapt_to_errors and len(self.sigf) != nsigf:
            if self.minimize_mem:
                self.sigf = numpy.empty(nsigf, numpy.float_)
            else:
                self.sigf = numpy.ones(nsigf, numpy.float_)
                self.sum_sigf = nsigf  
        self.neval_hcube = numpy.empty(self.nhcube_batch, dtype=numpy.intp)
        self.neval_hcube[:] = avg_neval_hcube
        self.y = numpy.empty((neval_batch, self.dim), numpy.float_)
        self.x = numpy.empty((neval_batch, self.dim), numpy.float_)
        self.jac = numpy.empty(neval_batch, numpy.float_)
        self.fdv2 = numpy.empty(neval_batch, numpy.float_)
        return old_val

    def settings(Integrator self not None, ngrid=0):
        """ Assemble summary of integrator settings into string.

        Args:
            ngrid (int): Number of grid nodes in each direction
                to include in summary.
                The default is 0.
        Returns:
            String containing the settings.
        """
        cdef numpy.npy_intp d
        nhcube = numpy.product(self.nstrat)
        neval = nhcube * self.min_neval_hcube if self.beta <= 0 else self.neval
        ans = ""
        ans = "Integrator Settings:\n"
        if self.beta > 0 and not self.adapt_to_errors:
            ans = ans + (
                "    %.6g (approx) integrand evaluations in each of %d iterations\n"
                % (self.neval, self.nitn)
                )
        else:
            ans = ans + (
                "    %.6g integrand evaluations in each of %d iterations\n"
                % (neval, self.nitn)
                )
        ans = ans + (
            "    number of:  strata/axis = %s\n" % str(numpy.array(self.nstrat))
            )
        ans = ans + (
            "                increments/axis = %s\n" 
            % str(numpy.asarray(self.map.ninc))
            )
        if self.beta > 0 and not self.adapt_to_errors:
            ans = ans + (
                "                h-cubes = %.6g  evaluations/h-cube = %d (min)\n"
                % (nhcube, self.min_neval_hcube)
                )
        else:
            ans = ans + (
                    "                h-cubes = %.6g evaluations/h-cube = %d\n"
                    % (nhcube, self.min_neval_hcube)
                    )
        ans += "                h-cubes/batch = %d\n" % self.nhcube_batch
        ans = ans + (
            "    minimize_mem = %s\n"
            % ('True' if self.minimize_mem else 'False')
            )
        ans = ans + (
            "    adapt_to_errors = %s\n"
            % ('True' if self.adapt_to_errors else 'False')
            )
        if not self.adapt:
            ans = ans + (
                "    damping parameters: alpha = %g  beta= %g\n"
                % (0., 0.)
                )
        elif self.adapt_to_errors:
            ans = ans + (
                "    damping parameters: alpha = %g  beta= %g\n"
                % (self.alpha, 0.) 
                )
        else:
            ans = ans + (
                "    damping parameters: alpha = %g  beta= %g\n"
                % (self.alpha, self.beta)
                )
        max_neval_hcube = max(self.max_neval_hcube, self.min_neval_hcube)
        ans += (
            "    limits: h-cubes < %.2g  evaluations/h-cube < %.2g\n"
            % (float(self.max_nhcube), float(max_neval_hcube))
            )
        ans = ans + ("    accuracy: relative = %g" % self.rtol)
        ans = ans + ("  absolute accuracy = %g\n\n" % self.atol)
        for d in range(self.dim):
            ans = ans +(
                "    axis %d covers %s\n" % (d, str(self.map.region(d)))
                )
        if ngrid > 0:
            ans += '\n' + self.map.settings(ngrid=ngrid)
        return ans

    def _fill_sigf(
        Integrator self not None, fcn,  numpy.npy_intp hcube_base, numpy.npy_intp nhcube_batch
        ):
        cdef numpy.npy_intp i_start
        cdef numpy.npy_intp ihcube, hcube, tmp_hcube
        cdef numpy.npy_intp[::1] y0 = numpy.empty(self.dim, numpy.intp)
        cdef numpy.npy_intp i, d
        cdef numpy.npy_intp neval_hcube = self.min_neval_hcube
        cdef numpy.npy_intp neval_batch = nhcube_batch * neval_hcube
        cdef double[:, ::1] yran = self.ran_array_generator((neval_batch, self.dim))
        cdef double dv_y = 1. / numpy.product(self.nstrat)
        cdef double sum_fdv, sum_fdv2, sigf2, fdv
        cdef numpy.ndarray[numpy.double_t, ndim=1] fx
        cdef double[:, ::1] y = self.y[:neval_batch, :]
        cdef double[:, ::1] x = self.x[:neval_batch, :]
        cdef double[::1] jac = self.jac[:neval_batch]
        cdef double[::1] sigf = self.sigf
        # generate random points
        i_start = 0
        for ihcube in range(nhcube_batch):
            hcube = hcube_base + ihcube
            tmp_hcube = hcube
            for d in range(self.dim):
                y0[d] = tmp_hcube % self.nstrat[d]
                tmp_hcube = (tmp_hcube - y0[d]) // self.nstrat[d]
            for d in range(self.dim):
                for i in range(i_start, i_start + neval_hcube):
                    y[i, d] = (y0[d] + yran[i, d]) / self.nstrat[d]
            i_start += neval_hcube
        self.map.map(y, x, jac, neval_batch)
        fx = fcn.training(numpy.asarray(x))

        # accumulate sigf for each h-cube
        i_start = 0
        for ihcube in range(nhcube_batch):
            sum_fdv = 0.0
            sum_fdv2 = 0.0
            for i in range(i_start, i_start + neval_hcube):
                fdv = fx[i] * jac[i] * dv_y
                sum_fdv += fdv
                sum_fdv2 += fdv ** 2
            mean = sum_fdv / neval_hcube
            sigf2 = abs(sum_fdv2 / neval_hcube - mean * mean)
            sigf[ihcube] = sigf2 ** (self.beta / 2.)
            i_start += neval_hcube

    def _get_mpi_rank(self):
        return (
            0 if (mpi4py is None or not self.mpi) else
            mpi4py.MPI.COMM_WORLD.Get_rank()
            )

    mpi_rank = property(_get_mpi_rank, doc="MPI rank (>=0)")

    def random_batch(
        Integrator self not None,
        bint yield_hcube=False,
        bint yield_y=False,
        fcn = None,
        ):
        """ Iterator over integration points and weights.

        This method creates an iterator that returns integration
        points from |vegas|, and their corresponding weights in an
        integral. The points are provided in arrays ``x[i, d]`` where
        ``i=0...`` labels the integration points in a batch
        and ``d=0...`` labels direction. The corresponding
        weights assigned by |vegas| to each point are provided
        in an array ``wgt[i]``.

        Optionally the integrator will also return the indices of
        the hypercubes containing the integration points and/or the |y|-space
        coordinates of those points::

            integ.random_batch()  yields  x, wgt

            integ.random_batch(yield_hcube=True) yields x, wgt, hcube

            integ.random_batch(yield_y=True) yields x, y, wgt

            integ.random_batch(yield_hcube=True, yield_y=True) yields x, y, wgt, hcube

        The number of integration points returned by the iterator
        corresponds to a single iteration. The number in a batch
        is controlled by parameter ``nhcube_batch``.
        """
        cdef numpy.npy_intp nhcube = numpy.product(self.nstrat)
        cdef double dv_y = 1. / nhcube
        cdef numpy.npy_intp nhcube_batch = min(self.nhcube_batch, nhcube)
        cdef numpy.npy_intp neval_batch
        cdef numpy.npy_intp hcube_base
        cdef numpy.npy_intp i_start, ihcube, i, d, tmp_hcube, hcube
        cdef numpy.npy_intp[::1] hcube_array
        cdef double neval_sigf = (
            self.neval_frac * self.neval / self.sum_sigf
            if self.beta > 0 and self.sum_sigf > 0 and not self.adapt_to_errors
            else 0.0    # use min_neval_hcube
            )
        cdef numpy.npy_intp avg_neval_hcube = int(self.neval / self.nhcube) ####
        cdef numpy.npy_intp[::1] neval_hcube = self.neval_hcube
        cdef numpy.npy_intp[::1] y0 = numpy.empty(self.dim, numpy.intp)
        cdef numpy.npy_intp max_neval_hcube = max(
            self.max_neval_hcube, self.min_neval_hcube
            )
        cdef double[::1] sigf
        cdef double[:, ::1] yran
        cdef double[:, ::1] y
        cdef double[:, ::1] x
        cdef double[::1] jac
        cdef bint adaptive_strat = (self.beta > 0 and nhcube > 1 and not self.adapt_to_errors)
        self.last_neval = 0
        self.neval_hcube_range = numpy.zeros(2, numpy.intp) + self.min_neval_hcube
        if yield_hcube:
            hcube_array = numpy.empty(self.y.shape[0], numpy.intp)
        if adaptive_strat and self.minimize_mem and not self.adapt:
            # can't minimize_mem without also adapting, so force beta=0
            neval_sigf = 0.0
        for hcube_base in range(0, nhcube, nhcube_batch):
            if (hcube_base + nhcube_batch) > nhcube:
                nhcube_batch = nhcube - hcube_base

            # determine number of evaluations per h-cube
            if adaptive_strat:
                if self.minimize_mem:
                    if self.adapt:
                        self._fill_sigf(
                            fcn=fcn, hcube_base=hcube_base, nhcube_batch=nhcube_batch,
                            )
                    sigf = self.sigf
                else:
                    sigf = self.sigf[hcube_base:]
                neval_batch = 0
                for ihcube in range(nhcube_batch):
                    neval_hcube[ihcube] = <int> (sigf[ihcube] * neval_sigf) + self.min_neval_hcube
                    if neval_hcube[ihcube] > max_neval_hcube:
                        neval_hcube[ihcube] = max_neval_hcube
                    if neval_hcube[ihcube] < self.neval_hcube_range[0]:
                        self.neval_hcube_range[0] = neval_hcube[ihcube]
                    elif neval_hcube[ihcube] > self.neval_hcube_range[1]:
                        self.neval_hcube_range[1] = neval_hcube[ihcube]
                    neval_batch += neval_hcube[ihcube]
            else:
                neval_hcube[:] = avg_neval_hcube
                neval_batch = nhcube_batch * avg_neval_hcube
            self.last_neval += neval_batch

            if (3*self.dim + 3) * neval_batch > self.max_mem:
                raise MemoryError('work arrays too large; reduce max_neval_hcube or nhcube_batch (or increase max_mem)')

            # resize work arrays if needed
            if neval_batch > self.y.shape[0]:
                self.y = numpy.empty((neval_batch, self.dim), numpy.float_)
                self.x = numpy.empty((neval_batch, self.dim), numpy.float_)
                self.jac = numpy.empty(neval_batch, numpy.float_)
                self.fdv2 = numpy.empty(neval_batch, numpy.float_)
            y = self.y
            x = self.x
            jac = self.jac
            if yield_hcube and neval_batch > hcube_array.shape[0]:
                hcube_array = numpy.empty(neval_batch, numpy.intp)

            # generate random points
            yran = self.ran_array_generator((neval_batch, self.dim))
            i_start = 0
            for ihcube in range(nhcube_batch):
                hcube = hcube_base + ihcube
                tmp_hcube = hcube
                for d in range(self.dim):
                    y0[d] = tmp_hcube % self.nstrat[d]
                    tmp_hcube = (tmp_hcube - y0[d]) // self.nstrat[d]
                for d in range(self.dim):
                    for i in range(i_start, i_start + neval_hcube[ihcube]):
                        y[i, d] = (y0[d] + yran[i, d]) / self.nstrat[d]
                i_start += neval_hcube[ihcube]
            self.map.map(y, x, jac, neval_batch)

            # compute weights and yield answers
            i_start = 0
            for ihcube in range(nhcube_batch):
                for i in range(i_start, i_start + neval_hcube[ihcube]):
                    jac[i] *= dv_y / neval_hcube[ihcube]
                    if yield_hcube:
                        hcube_array[i] = hcube_base + ihcube
                i_start += neval_hcube[ihcube]
            answer = (numpy.asarray(x[:neval_batch, :]),)
            if yield_y:
                answer += (numpy.asarray(y[:neval_batch, :]),)
            answer += (numpy.asarray(jac[:neval_batch]),)
            if yield_hcube:
                answer += (numpy.asarray(hcube_array[:neval_batch]),)
            yield answer

    # old name --- for legacy code
    random_vec = random_batch

    def random(
        Integrator self not None, bint yield_hcube=False, bint yield_y=False
        ):
        """ Iterator over integration points and weights.

        This method creates an iterator that returns integration
        points from |vegas|, and their corresponding weights in an
        integral. Each point ``x[d]`` is accompanied by the weight
        assigned to that point by |vegas| when estimating an integral.
        Optionally it will also return the index of the hypercube
        containing the integration point and/or the |y|-space
        coordinates::

            integ.random()  yields  x, wgt

            integ.random(yield_hcube=True) yields x, wgt, hcube

            integ.random(yield_y=True) yields x, y, wgt

            integ.random(yield_hcube=True, yield_y=True) yields x, y, wgt, hcube

        The number of integration points returned by the iterator
        corresponds to a single iteration.
        """
        cdef double[:, ::1] x
        cdef double[::1] wgt
        cdef numpy.npy_intp[::1] hcube
        cdef double[:, ::1] y
        cdef numpy.npy_intp i
        if yield_hcube and yield_y:
            for x, y, wgt, hcube in self.random_batch(yield_hcube=True, yield_y=True):
                for i in range(x.shape[0]):
                    yield (x[i], y[i], wgt[i], hcube[i])
        elif yield_y:
            for x, y, wgt in self.random_batch(yield_y=True):
                for i in range(x.shape[0]):
                    yield (x[i], y[i], wgt[i])
        elif yield_hcube:
            for x, wgt, hcube in self.random_batch(yield_hcube=True):
                for i in range(x.shape[0]):
                    yield (x[i], wgt[i], hcube[i])
        else:
            for x,wgt in self.random_batch():
                for i in range(x.shape[0]):
                    yield (x[i], wgt[i])

    @staticmethod
    def synchronize_random():
        if mpi4py is None:
            return
        comm = mpi4py.MPI.COMM_WORLD
        rank = comm.Get_rank()
        nproc = comm.Get_size()
        if nproc > 1:
            # synchronize random numbers
            if rank == 0:
                seed = tuple(
                    numpy.random.randint(1, min(2**30, sys.maxsize), size=5)
                    )
            else:
                seed = None
            seed = comm.bcast(seed, root=0)
            numpy.random.seed(seed)

    def __call__(Integrator self not None, fcn, **kargs):
        """ Integrate integrand ``fcn``.

        A typical integrand has the form, for example::

            def f(x):
                return x[0] ** 2 + x[1] ** 4

        The argument ``x[d]`` is an integration point, where
        index ``d=0...`` represents direction within the
        integration volume.

        Integrands can be array-valued, representing multiple
        integrands: e.g., ::

            def f(x):
                return [x[0] ** 2, x[0] / x[1]]

        The return arrays can have any shape. Dictionary-valued
        integrands are also supported: e.g., ::

            def f(x):
                return dict(a=x[0] ** 2, b=[x[0] / x[1], x[1] / x[0]])

        Integrand functions that return arrays or dictionaries
        are useful for multiple integrands that are closely related,
        and can lead to substantial reductions in the errors for
        ratios or differences of the results.

        It is usually much faster to use |vegas| in batch
        mode, where integration points are presented to the
        integrand in batches. A simple batch integrand might
        be, for example::

            @vegas.batchintegrand
            def f(x):
                return x[:, 0] ** 2 + x[:, 1] ** 4

        where decorator ``@vegas.batchintegrand`` tells
        |vegas| that the integrand processes integration
        points in batches. The array ``x[i, d]``
        represents a collection of different integration
        points labeled by ``i=0...``. (The number of points is controlled
        |Integrator| parameter ``nhcube_batch``.) The batch index
        is always first.

        Batch integrands can also be constructed from classes
        derived from :class:`vegas.BatchIntegrand`.

        Batch mode is particularly useful (and fast) when the class
        derived from :class:`vegas.BatchIntegrand` is coded
        in Cython. Then loops over the integration points
        can be coded explicitly, avoiding the need to use
        :mod:`numpy`'s whole-array operators if they are not
        well suited to the integrand.

        Any |vegas| parameter can also be reset: e.g.,
        ``self(fcn, nitn=20, neval=1e6)``.
        """
        cdef numpy.ndarray[numpy.double_t, ndim=2] x
        cdef numpy.ndarray[numpy.double_t, ndim=1] wgt
        cdef numpy.ndarray[numpy.npy_intp, ndim=1] hcube
        cdef double[::1] sigf
        cdef double[:, ::1] y
        cdef double[::1] fdv2
        cdef double[:, ::1] fx
        cdef double[::1] wf
        cdef double[::1] sum_wf
        cdef double[:, ::1] sum_wf2
        cdef double[::1] mean = numpy.empty(1, numpy.float_)
        cdef double[:, ::1] var = numpy.empty((1, 1), numpy.float_)
        cdef double[::1] dvar = numpy.empty(1, numpy.float_)
        cdef numpy.npy_intp itn, i, j, s, t, neval 
        cdef double sum_sigf, sigf2
        cdef bint firsteval = True

        if kargs:
            self.set(kargs)

        # synchronize random numbers across all processes (mpi)
        if self.sync_ran and self.mpi:
            self.synchronize_random()

        # Put integrand into standard form
        fcn = VegasIntegrand(fcn, mpi=self.mpi)

        sigf = self.sigf
        for itn in range(self.nitn):
            # if self.minimize_mem:
            #     self.set()
            if self.analyzer is not None:
                self.analyzer.begin(itn, self)

            # initalize arrays that accumulate results for a single iteration
            mean[:] = 0.0
            var[:, :] = 0.0
            sum_sigf = 0.0

            # iterate batch-slices of integration points
            for x, y, wgt, hcube in self.random_batch(
                yield_hcube=True, yield_y=True, fcn=fcn
                ):
                fdv2 = self.fdv2        # must be inside loop

                # evaluate integrand at all points in x
                fx = fcn.eval(x)
                if firsteval:
                    # allocate work arrays on first pass through;
                    # (needed a sample fcn evaluation in order to do this)
                    firsteval = False
                    wf = numpy.empty(fcn.size, numpy.float_)
                    sum_wf = numpy.empty(fcn.size, numpy.float_)
                    sum_wf2 = numpy.empty((fcn.size, fcn.size), numpy.float_)
                    mean = numpy.empty(fcn.size, numpy.float_)
                    var = numpy.empty((fcn.size, fcn.size), numpy.float_)
                    dvar = numpy.empty(fcn.size, numpy.float_)
                    mean[:] = 0.0
                    var[:, :] = 0.0
                    result = VegasResult(fcn, weighted=self.adapt)

                # compute integral and variance for each h-cube
                # j is index of point within batch, i is hcube index
                j = 0
                for i in range(hcube[0], hcube[-1] + 1):
                    # iterate over h-cubes
                    sum_wf[:] = 0.0
                    sum_wf2[:, :] = 0.0
                    neval = 0
                    while j < len(hcube) and hcube[j] == i:
                        for s in range(fcn.size):
                            wf[s] = wgt[j] * fx[j, s]
                            sum_wf[s] += wf[s]
                            for t in range(s + 1):
                                sum_wf2[s, t] += wf[s] * wf[t]
                        fdv2[j] = (wf[0] * self.neval_hcube[i - hcube[0]]) ** 2
                        j += 1
                        neval += 1
                    for s in range(fcn.size):
                        mean[s] += sum_wf[s]
                        for t in range(s + 1):
                            dvar[t] = (
                                sum_wf2[s, t] * neval - sum_wf[s] * sum_wf[t]
                                ) / (neval - 1.)
                        if EPSILON * sum_wf2[s, s] > dvar[s]:
                            # roundoff error ==> add only on diagonal (uncorrelated)
                            var[s, s] += EPSILON * sum_wf2[s, s]
                        else:
                            for t in range(s + 1):
                                var[s, t] += dvar[t]
                    sigf2 = abs(sum_wf2[0, 0] * neval - sum_wf[0] * sum_wf[0])
                    if self.beta > 0 and self.adapt and not self.adapt_to_errors:
                        if not self.minimize_mem:
                            sigf[i] = sigf2 ** (self.beta / 2.)
                            sum_sigf += sigf[i]
                        else:
                            sum_sigf += sigf2 ** (self.beta / 2.)
                    if self.adapt_to_errors and self.adapt:
                        # replace fdv2 with variance
                        # only one piece of data (from current hcube)
                        fdv2[j - 1] = sigf2
                        self.map.add_training_data(
                            y[j - 1:, :], fdv2[j - 1:], 1
                            )
                if (not self.adapt_to_errors) and self.adapt and self.alpha > 0:
                    self.map.add_training_data(y, fdv2, y.shape[0])

            for s in range(var.shape[0]):
                for t in range(s):
                    var[t, s] = var[s, t]

            # accumulate result from this iteration
            result.update(mean, var, self.last_neval)

            if self.beta > 0 and not self.adapt_to_errors and self.adapt:
                self.sum_sigf = sum_sigf
            if self.alpha > 0 and self.adapt:
                self.map.adapt(alpha=self.alpha)
            if self.analyzer is not None:
                result.update_analyzer(self.analyzer)

            if result.converged(self.rtol, self.atol):
                break
        return result.result

class reporter:
    """ Analyzer class that prints out a report, iteration
    by interation, on how vegas is doing. Parameter ngrid
    specifies how many x[i]'s to print out from the maps
    for each axis.

    Args:
        ngrid (int): Number of grid nodes printed out for
            each direction. Default is 0.
    """
    def __init__(self, ngrid=0):
        self.ngrid = ngrid
        self.clock = time.process_time if hasattr(time, 'process_time') else time.time

    def begin(self, itn, integrator):
        self.integrator = integrator
        self.itn = itn
        self.t0 = self.clock()
        if itn==0:
            print(integrator.settings())
        sys.stdout.flush()

    def end(self, itn_ans, ans):
        print("    itn %2d: %s\n all itn's: %s"%(self.itn+1, itn_ans, ans))
        print(
            '    neval = %s  neval/h-cube = %s\n    chi2/dof = %.2f  Q = %.2f  time = %.2f'
            % (
                format(self.integrator.last_neval, '.6g'),
                tuple(self.integrator.neval_hcube_range),
                ans.chi2 / ans.dof if ans.dof > 0 else 0,
                ans.Q if ans.dof > 0 else 1.,
                self.clock() - self.t0
                )
            )
        print(self.integrator.map.settings(ngrid=self.ngrid))
        print('')
        sys.stdout.flush()

# Objects for accumulating the results from multiple iterations of vegas.
# Results can be scalars (RAvg), arrays (RAvgArray), or dictionaries (RAvgDict).
# Each stores results from each iterations, as well as a weighted (running)
# average of the results of all iterations (unless parameter weigthed=False,
# in which case the average is unweighted).
class RAvg(gvar.GVar):
    """ Running average of scalar-valued Monte Carlo estimates.

    This class accumulates independent Monte Carlo
    estimates (e.g., of an integral) and combines
    them into a single average. It
    is derived from :class:`gvar.GVar` (from
    the :mod:`gvar` module if it is present) and
    represents a Gaussian random variable.

    Different estimates are weighted by their
    inverse variances if parameter ``weight=True``;
    otherwise straight, unweighted averages are used.
    """
    def __init__(self, weighted=True):
        if weighted:
            self._v_s2 = 0.
            self._v2_s2 = 0.
            self._1_s2 = 0.
            self.weighted = True
        else:
            self._v = 0.
            self._v2 = 0.
            self._s2 = 0.
            self._n = 0
            self.weighted = False
        self.itn_results = []
        super(RAvg, self).__init__(
            *gvar.gvar(0., 0.).internaldata,
           )

    def _chi2(self):
        if len(self.itn_results) <= 1:
            return 0.0
        if self.weighted:
            ans = self._v2_s2 - self._v_s2 ** 2 / self._1_s2
            if ans < 0 or self._v2_s2 * EPSILON > ans:
                ans = float('nan')
            return ans 
        else:
            ans = (self._v2 - self.mean ** 2 * self._n) * self._n / self._s2
            if ans < 0 or self._v2 * EPSILON > ans * self._s2:
                ans = float('nan')
            return ans
    chi2 = property(_chi2, None, None, "*chi**2* of weighted average.")

    def _dof(self):
        return len(self.itn_results) - 1
    dof = property(
        _dof,
        None,
        None,
        "Number of degrees of freedom in weighted average."
        )

    def _nitn(self):
        return len(self.itn_results)

    nitn = property(_nitn, None, None, "Number of iterations.")

    def _Q(self):
        return (
            gvar.gammaQ(self.dof / 2., self.chi2 / 2.)
            if self.dof > 0 and self.chi2 > 0
            else float('nan')
            )
    Q = property(
        _Q,
        None,
        None,
        "*Q* or *p-value* of weighted average's *chi**2*.",
        )

    def converged(self, rtol, atol):
        return self.sdev < atol + rtol * abs(self.mean)

    def add(self, g):
        """ Add estimate ``g`` to the running average. """
        self.itn_results.append(g)
        if self.weighted:
            var = g.sdev ** 2 if g.sdev > 0 else TINY
            if var < TINY:
                var = TINY
            self._v_s2 +=   g.mean / var
            self._v2_s2 +=  g.mean ** 2 / var
            self._1_s2 +=  1. / var
            super(RAvg, self).__init__(*gvar.gvar(
                self._v_s2 / self._1_s2,
                sqrt(1. / self._1_s2),
                ).internaldata)
        else:
            self._v += g.mean
            self._v2 += g.mean ** 2
            self._s2 += g.var if g.var > 0 else TINY
            self._n += 1
            super(RAvg, self).__init__(*gvar.gvar(
                self._v / self._n,
                sqrt(self._s2) / self._n,
                ).internaldata)


    def summary(self, extended=False, weighted=None):
        """ Assemble summary of results, iteration-by-iteration, into a string.

        Args:
            weighted (bool): Display weighted averages of results from different
                iterations if ``True``; otherwise show unweighted averages.
                Default behavior is determined by |vegas|.
        """
        if weighted is None:
            weighted = self.weighted
        acc = RAvg(weighted=weighted)
        linedata = []
        for i, res in enumerate(self.itn_results):
            acc.add(res)
            if i > 0:
                chi2_dof = acc.chi2 / acc.dof
                Q = acc.Q
            else:
                chi2_dof = 0.0
                Q = 1.0
            itn = '%3d' % (i + 1)
            integral = '%-15s' % res
            wgtavg = '%-15s' % acc
            chi2dof = '%8.2f' % (acc.chi2 / acc.dof if i != 0 else 0.0)
            Q = '%8.2f' % (acc.Q if i != 0 else 1.0)
            linedata.append((itn, integral, wgtavg, chi2dof, Q))
        nchar = 5 * [0]
        for data in linedata:
            for i, d in enumerate(data):
                if len(d) > nchar[i]:
                    nchar[i] = len(d)
        fmt = '%%%ds   %%-%ds %%-%ds %%%ds %%%ds\n' % tuple(nchar)
        if weighted:
            ans = fmt % ('itn', 'integral', 'wgt average', 'chi2/dof', 'Q')
        else:
            ans = fmt % ('itn', 'integral', 'average', 'chi2/dof', 'Q')
        ans += len(ans[:-1]) * '-' + '\n'
        for data in linedata:
            ans += fmt % data
        return ans

class RAvgDict(gvar.BufferDict):
    """ Running average of dictionary-valued Monte Carlo estimates.

    This class accumulates independent dictionaries of Monte Carlo
    estimates (e.g., of an integral) and combines
    them into a dictionary of averages. It
    is derived from :class:`gvar.BufferDict`. The dictionary
    values are :class:`gvar.GVar`\s or arrays of :class:`gvar.GVar`\s.

    Different estimates are weighted by their
    inverse covariance matrices if parameter ``weight=True``;
    otherwise straight, unweighted averages are used.
    """
    def __init__(self, dictionary, weighted=True):
        super(RAvgDict, self).__init__(dictionary)
        self.rarray = RAvgArray(shape=(self.size,), weighted=weighted)
        self.buf = numpy.array(self.rarray)
        self.itn_results = []
        self.weighted = weighted

    def converged(self, rtol, atol):
        return numpy.all(
            gvar.sdev(self.buf) <
            atol + rtol * numpy.abs(gvar.mean(self.buf))
            )

    def add(self, g):
        if isinstance(g, gvar.BufferDict):
            newg = gvar.BufferDict(g)
        else:
            newg = gvar.BufferDict()
            for k in self:
                try:
                    newg[k] = g[k]
                except AttributeError:
                    raise ValueError(
                        "Dictionary g doesn't contain key " + str(k) + '.'
                        )
        self.itn_results.append(newg)
        self.rarray.add(newg.buf)
        self.buf = numpy.array(self.rarray)

    def summary(self, extended=False, weighted=None):
        """ Assemble summary of results, iteration-by-iteration, into a string.

        Args:
            extended (bool): Include a table of final averages for every
                component of the integrand if ``True``. Default is ``False``.
            weighted (bool): Display weighted averages of results from different
                iterations if ``True``; otherwise show unweighted averages.
                Default behavior is determined by |vegas|.
        """
        if weighted is None:
            weighted = self.weighted
        ans = self.rarray.summary(weighted=weighted, extended=False)
        if extended and self.itn_results[0].size > 1:
            ans += '\n' + gvar.tabulate(self)
        return ans

    def _chi2(self):
        return self.rarray.chi2
    chi2 = property(_chi2, None, None, "*chi**2* of weighted average.")

    def _dof(self):
        return self.rarray.dof
    dof = property(
        _dof, None, None,
        "Number of degrees of freedom in weighted average."
        )

    def _nitn(self):
        return len(self.itn_results)

    nitn = property(_nitn, None, None, "Number of iterations.")

    def _Q(self):
        return self.rarray.Q
    Q = property(
        _Q, None, None,
        "*Q* or *p-value* of weighted average's *chi**2*.",
        )


class RAvgArray(numpy.ndarray):
    """ Running average of array-valued Monte Carlo estimates.

    This class accumulates independent arrays of Monte Carlo
    estimates (e.g., of an integral) and combines
    them into an array of averages. It
    is derived from :class:`numpy.ndarray`. The array
    elements are :class:`gvar.GVar`\s (from the ``gvar`` module if
    present) and represent Gaussian random variables.

    Different estimates are weighted by their
    inverse covariance matrices if parameter ``weight=True``;
    otherwise straight, unweighted averages are used.
    """
    def __new__(
        subtype, shape, weighted=True,
        dtype=object, buffer=None, offset=0, strides=None, order=None
        ):
        obj = numpy.ndarray.__new__(
            subtype, shape=shape, dtype=object, buffer=buffer, offset=offset,
            strides=strides, order=order
            )
        if buffer is None:
            obj.flat = numpy.array(obj.size * [gvar.gvar(0,0)])
        obj.itn_results = []
        if weighted:
            obj._invcov_v = 0.
            obj._v_invcov_v = 0.
            obj._invcov = 0.
            obj.weighted = True
        else:
            obj._v = 0.
            obj._v2 = 0.
            obj._cov = 0.
            obj._n = 0
            obj.weighted = False
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.itn_results = getattr(obj, 'itn_results', [])
        if obj.weighted:
            self._invcov_v = getattr(obj, '_invcov_v', 0.0)
            self._v_invcov_v = getattr(obj, '_v_invcov_v', 0.0)
            self._invcov = getattr(obj, '_invcov', 0.0)
            self.weighted = getattr(obj, 'weighted', True)
        else:
            self._v = getattr(obj, '_v', 0.)
            self._v2 = getattr(obj, '_v2', 0.)
            self._cov = getattr(obj, '_cov', 0.)
            self._n = getattr(obj, '_n', 0.)
            self.weighted = getattr(obj, 'weighted', False)

    def _inv(self, matrix):
        " Invert matrix, with protection against singular matrices. "
        # old code -- problem is not necessarily pos. def.
        # ans = numpy.linalg.pinv(matrix, rcond=EPSILON * len(matrix))
        # return (ans.T + ans) / 2.
        #
        # rescale = False is important for deg. multi-integrands.
        # svdcut<0 also important for deg. multi-integrands.
        s = gvar.SVD(matrix, svdcut=-EPSILON * len(matrix), rescale=False)
        w = s.decomp(-1)
        return (w.T).dot(w)

    def converged(self, rtol, atol):
        return numpy.all(
            gvar.sdev(self) < atol + rtol * numpy.abs(gvar.mean(self))
            )

    def _chi2(self):
        if len(self.itn_results) <= 1:
            return 0.0
        if self.weighted:
            cov = self._inv(self._invcov)
            ans = self._v_invcov_v - self._invcov_v.dot(cov.dot(self._invcov_v))
            if ans < 0 or self._v_invcov_v * EPSILON > ans:
                ans = float('nan')
            return ans
        else:
            invcov = self._inv(self._cov / self._n)
            ans = numpy.trace(   # inefficient -- fix at some point
                (self._v2 - numpy.outer(self._v, self._v) / self._n).dot(invcov)
                )
            if ans < 0 or numpy.trace(self._v2.dot(invcov)) * EPSILON > ans:
                ans = float('nan')
            return ans
    chi2 = property(_chi2, None, None, "*chi**2* of weighted average.")

    def _dof(self):
        if len(self.itn_results) <= 1:
            return 0
        return (len(self.itn_results) - 1) * self.itn_results[0].size
    dof = property(
        _dof, None, None,
        "Number of degrees of freedom in weighted average."
        )

    def _nitn(self):
        return len(self.itn_results)

    nitn = property(_nitn, None, None, "Number of iterations.")

    def _Q(self):
        if self.dof <= 0 or self.chi2 <= 0:
            return float('nan')
        return gvar.gammaQ(self.dof / 2., self.chi2 / 2.)
    Q = property(
        _Q, None, None,
        "*Q* or *p-value* of weighted average's *chi**2*.",
        )

    def add(self, g):
        """ Add estimate ``g`` to the running average. """
        g = numpy.asarray(g)
        self.itn_results.append(g)
        g = g.reshape((-1,))
        gmean = gvar.mean(g)
        gcov = gvar.evalcov(g)
        idx = (gcov[numpy.diag_indices_from(gcov)] <= 0.0)
        gcov[numpy.diag_indices_from(gcov)][idx] = TINY
        if self.weighted:
            invcov = self._inv(gcov)
            v = gvar.mean(g)
            u = invcov.dot(v)
            self._invcov += invcov
            self._invcov_v += u
            self._v_invcov_v += v.dot(u)
            cov = self._inv(self._invcov)
            mean = cov.dot(self._invcov_v)
            self[:] = gvar.gvar(mean, cov).reshape(self.shape)
        else:
            gmean = gvar.mean(g)
            self._v2 += numpy.outer(gmean, gmean)
            self._v += gmean
            self._cov += gcov
            self._n += 1
            mean = self._v / self._n
            cov = self._cov / (self._n ** 2)
            self[:] = gvar.gvar(mean, cov).reshape(self.shape)

    def summary(self, extended=False, weighted=None):
        """ Assemble summary of results, iteration-by-iteration, into a string.

        Args:
            extended (bool): Include a table of final averages for every
                component of the integrand if ``True``. Default is ``False``.
            weighted (bool): Display weighted averages of results from different
                iterations if ``True``; otherwise show unweighted averages.
                Default behavior is determined by |vegas|.
        """
        if weighted is None:
            weighted = self.weighted
        acc = RAvgArray(self.shape, weighted=weighted)

        linedata = []
        for i, res in enumerate(self.itn_results):
            acc.add(res)
            if i > 0:
                chi2_dof = acc.chi2 / acc.dof
                Q = acc.Q
            else:
                chi2_dof = 0.0
                Q = 1.0
            itn = '%3d' % (i + 1)
            integral = '%-15s' % res.flat[0]
            wgtavg = '%-15s' % acc.flat[0]
            chi2dof = '%8.2f' % (acc.chi2 / acc.dof if i != 0 else 0.0)
            Q = '%8.2f' % (acc.Q if i != 0 else 1.0)
            linedata.append((itn, integral, wgtavg, chi2dof, Q))
        nchar = 5 * [0]
        for data in linedata:
            for i, d in enumerate(data):
                if len(d) > nchar[i]:
                    nchar[i] = len(d)
        fmt = '%%%ds   %%-%ds %%-%ds %%%ds %%%ds\n' % tuple(nchar)
        if weighted:
            ans = fmt % ('itn', 'integral', 'wgt average', 'chi2/dof', 'Q')
        else:
            ans = fmt % ('itn', 'integral', 'average', 'chi2/dof', 'Q')
        ans += len(ans[:-1]) * '-' + '\n'
        for data in linedata:
            ans += fmt % data
        if extended and self.itn_results[0].size > 1:
            ans += '\n' + gvar.tabulate(self)
        return ans


################
# Classes that standarize the interface for integrands. Internally vegas
# assumes batch integrands that return an array fx[i, d] where i = batch index
# and d = index over integrand components. VegasIntegrand figures out how
# to convert the various types of integrand to this format. Integrands that
# return scalars or arrays or dictionaries lead to integration results that
# scalars or arrays or dictionaries, respectively; VegasResult figures
# out how to convert the 1-d array used internally in vegas into the
# appropriate structure given the integrand structure.

cdef class VegasResult:
    cdef readonly object integrand
    cdef readonly object shape
    cdef readonly object result
    cdef readonly double sum_neval
    """ Accumulated result object --- standard interface for integration results.

    Integrands are flattened into 2-d arrays in |vegas|. This object
    accumulates integration results from multiple iterations of |vegas|
    and can convert them to the original integrand format. It also counts
    the number of integrand evaluations used in all and adds it to the 
    result (``sum_neval``).

    Args:
        integrand: :class:`VegasIntegrand` object.
        weighted (bool): use weighted average across iterations?

    Attributes:
        shape: shape of integrand result or ``None`` if dictionary.
        result: accumulation of integral results. This is an object
            of type :class:`vegas.RAvgArray` for array-valued integrands,
            :class:`vegas.RAvgDict` for dictionary-valued integrands, and
            :class:`vegas.RAvg` for scalar-valued integrands.
        sum_neval: total number of integrand evaluations in all iterations.
        avg_neval: average number of integrand evaluations per iteration.
    """
    def __init__(self, integrand=None, weighted=None):
        self.integrand = integrand
        self.shape = integrand.shape
        self.sum_neval = 0
        if self.shape is None:
            self.result = RAvgDict(integrand.bdict, weighted=weighted)
        elif self.shape == ():
            self.result = RAvg(weighted=weighted)
        else:
            self.result = RAvgArray(self.shape, weighted=weighted)

    property  avg_neval:
        " Average number of integrand evaluations per iteration."
        def __get__(self):
            if len(self.result.itn_results) > 0:
                return int(self.sum_neval / len(self.result.itn_results))
            else:
                return 0.

    def update(self, mean, var, last_neval=None):
        if self.shape is None:
            ans = gvar.BufferDict(self.integrand.bdict, buf=gvar.gvar(mean, var))
            self.result.add(ans)
        elif self.shape == ():
            self.result.add(gvar.gvar(mean[0], var[0,0] ** 0.5))
        else:
            self.result.add(gvar.gvar(mean, var).reshape(self.shape))
        if last_neval is not None:
            self.sum_neval += last_neval
            self.result.sum_neval = self.sum_neval
            self.result.avg_neval = self.avg_neval

    def update_analyzer(self, analyzer):
        """ Update analyzer at end of an iteration. """
        analyzer.end(self.result.itn_results[-1], self.result)

    def converged(self, rtol, atol):
        " Convergence test. "
        return self.result.converged(rtol, atol)

cdef class VegasIntegrand:
    cdef public object shape
    cdef public numpy.npy_intp size
    cdef public object eval
    cdef public object bdict
    cdef public int nproc
    cdef public int rank
    cdef public object comm
    """ Integand object --- standard interface for integrands

    This class provides a standard interface for all |vegas| integrands.
    It analyzes the integrand to determine the shape of its output.

    All integrands are converted to batch integrands. Method ``eval(x)``
    returns results packed into a 2-d array ``f[i, d]`` where ``i``
    is the batch index and ``d`` indexes the different elements of the
    integrand.

    :class:`vegas.Integrand` doesn't know anything about the integrand
    until method ``eval(x)`` is called the first time. It then examines
    the result returned by the integrand function to figure out what
    kind of integrand it is dealing with. It waits until this point
    because it needs a valid ``x`` at which to evaluate the integrand.

    The integrands are automatically configured for parallel processing
    using MPI (via :mod:`mpi4py`).

    Args:
        fcn: Integrand function.
        map: :class:`vegas.AdaptiveMap` being used by |vegas|.

    Attributes:
        eval: ``eval(x)`` returns ``fcn(x)`` repacked as a 2-d array.
        shape: Shape of integrand ``fcn(x)`` or ``None`` if it is a dictionary.
        size: Size of integrand.
        nproc: Number of processors (=1 if no MPI)
        rank: MPI rank of processors (=0 if no MPI)
    """
    def __init__(self, fcn, mpi=True):
        if isinstance(fcn, type(BatchIntegrand)):
            raise ValueError(
                'integrand given is a class, not an object -- need parentheses?'
                )
        if mpi4py is None or not mpi:
            self.nproc = 1
        else:
            self.comm = mpi4py.MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.nproc = self.comm.Get_size()
        def eval(x, self=self, fcn=fcn):
            " Temporary eval, used for first call but then replaced by correct eval. "
            # check for scalar functions and convert to batch if is scalar
            # evaluate at arbitrary point to determine integrand shape
            x0 = x[0]
            fcntype = getattr(fcn, 'fcntype', 'scalar')
            if fcntype == 'scalar':
                fx = fcn(x0)
                if hasattr(fx, 'keys'):
                    if not isinstance(fx, gvar.BufferDict):
                        fx = gvar.BufferDict(fx)
                    self.size = fx.size
                    self.shape = None
                    self.bdict = fx
                    _eval = _BatchIntegrand_from_NonBatchDict(fcn, self.size)
                else:
                    fx = numpy.asarray(fcn(x0))
                    self.shape = fx.shape
                    self.size = fx.size
                    _eval = _BatchIntegrand_from_NonBatch(fcn, self.size, self.shape)
            else:
                x0.shape = (1,) + x0.shape
                fx = fcn(x0)
                if hasattr(fx, 'keys'):
                    # build dictionary for non-batch version of function
                    fxs = gvar.BufferDict()
                    for k in fx:
                        fxs[k] = fx[k][0]
                    self.shape = None
                    self.bdict = fxs
                    self.size = self.bdict.size
                    _eval = _BatchIntegrand_from_BatchDict(fcn, self.bdict)
                else:
                    fx = numpy.asarray(fcn(x0))
                    self.shape = fx.shape[1:]
                    self.size = fx.size
                    _eval = _BatchIntegrand_from_Batch(fcn, self.shape)
            if self.nproc > 1:
                # MPI multiprocessor mode
                def _mpi_eval(x, self=self, _eval=_eval):
                    nx = x.shape[0] // self.nproc + 1
                    i0 = self.rank * nx
                    i1 = min(i0 + nx, x.shape[0])
                    f = numpy.empty((nx, self.size), numpy.float_)
                    if i1 > i0:
                        # fill f so long as haven't gone off end
                        f[:(i1-i0)] = _eval(x[i0:i1])
                    results = numpy.empty((self.nproc * nx, self.size), numpy.float_)
                    self.comm.Allgather(f, results)
                    return results[:x.shape[0]]
                self.eval = _mpi_eval
            else:
                self.eval = _eval
            return self.eval(x)
        self.eval = eval

    def training(self, x):
        """ Calculate first element of integrand at point ``x``. """
        cdef numpy.ndarray fx =self.eval(x)
        if fx.ndim == 1:
            return fx
        else:
            fx = fx.reshape((x.shape[0], -1))
            return fx[:, 0]

# The _BatchIntegrand_from_XXXX objects are used by VegasIntegrand
# to convert different types of integrand (ie, scalar vs array vs dict,
# and nonbatch vs batch) to the standard output format assumed internally
# in vegas.
cdef class _BatchIntegrand_from_NonBatch(object):
    cdef readonly numpy.npy_intp size
    cdef readonly object shape
    cdef readonly object fcn
    """ Batch integrand from non-batch integrand. """
    def __init__(self, fcn, size, shape):
        self.fcn = fcn
        self.size = size
        self.shape = shape

    def __call__(self, numpy.ndarray[numpy.double_t, ndim=2] x):
        cdef numpy.npy_intp i
        cdef numpy.ndarray[numpy.float_t, ndim=2] f = numpy.empty(
            (x.shape[0], self.size),  numpy.float_
            )
        if self.shape == ():
            # very common special case
            for i in range(x.shape[0]):
                f[i] = self.fcn(x[i])
        else:
            for i in range(x.shape[0]):
                f[i] = numpy.asarray(self.fcn(x[i])).reshape((-1,))
        return f

cdef class _BatchIntegrand_from_NonBatchDict(object):
    cdef readonly numpy.npy_intp size
    cdef readonly object fcn
    """ Batch integrand from non-batch dict-integrand. """
    def __init__(self, fcn, size):
        self.fcn = fcn
        self.size = size

    def __call__(self, numpy.ndarray[numpy.double_t, ndim=2] x):
        cdef numpy.npy_intp i
        cdef numpy.ndarray[numpy.double_t, ndim=2] f = numpy.empty(
            (x.shape[0], self.size), float
            )
        for i in range(x.shape[0]):
            fx = self.fcn(x[i])
            if not isinstance(fx, gvar.BufferDict):
                fx = gvar.BufferDict(fx)
            f[i] = fx.buf
        return f

cdef class _BatchIntegrand_from_Batch(object):
    cdef readonly object fcn
    cdef readonly object shape
    """ BatchIntegrand from batch function. """
    def __init__(self, fcn, shape):
        self.fcn = fcn
        self.shape = shape

    def __call__(self, numpy.ndarray[numpy.double_t, ndim=2] x):
        if x.shape[0] <= 0:
            return numpy.empty((0,) + self.shape, float)
        fx = self.fcn(x)
        if not isinstance(fx, numpy.ndarray):
            fx = numpy.asarray(fx)
        return fx if len(fx.shape) == 2 else fx.reshape((x.shape[0], -1))

cdef class _BatchIntegrand_from_BatchDict(object):
    cdef readonly numpy.npy_intp size
    cdef readonly object slice
    cdef readonly object shape
    cdef object fcn
    """ BatchIntegrand from non-batch dict-integrand. """
    def __init__(self, fcn, bdict):
        self.fcn = fcn
        self.size = bdict.size
        self.slice = collections.OrderedDict()
        self.shape = collections.OrderedDict()
        for k in bdict:
            self.slice[k], self.shape[k] = bdict.slice_shape(k)

    def __call__(self, numpy.ndarray[numpy.double_t, ndim=2] x):
        cdef numpy.npy_intp i
        cdef numpy.ndarray[numpy.double_t, ndim=2] buf = numpy.empty(
            (x.shape[0], self.size), float
            )
        if x.shape[0] <= 0:
            return buf
        fx = self.fcn(x)
        for k in self.slice:
            buf[:, self.slice[k]] = (
                fx[k]
                if self.shape[k] is () else
                numpy.asarray(fx[k]).reshape((x.shape[0], -1))
                )
        return buf

# BatchIntegrand is a base class for users who want to design
# batch integrands.
cdef class BatchIntegrand:
    """ Base class for classes providing batch integrands.

    A class derived from :class:`vegas.BatchIntegrand` will normally
    provide a ``__call__(self, x)`` method that returns an
    array ``f`` where:

        ``x[i, d]`` is a contiguous :mod:`numpy` array where ``i=0...``
        labels different integrtion points and ``d=0...`` labels
        different directions in the integration space.

        ``f[i]`` is a contiguous array containing the integrand
        values corresponding to the integration
        points ``x[i, :]``. ``f[i]`` is either a number,
        for a single integrand, or an array (of any shape)
        for multiple integrands (i.e., an
        array-valued integrand).

    An example is ::

        import vegas
        import numpy as np

        class batchf(vegas.BatchIntegrand):
            def __call__(x):
                return np.exp(-x[:, 0] - x[:, 1])

        f = batchf()      # the integrand

    for the two-dimensional integrand :math:`\exp(-x_0 - x_1)`.

    Deriving from :class:`vegas.BatchIntegrand` is the
    easiest way to construct integrands in Cython, and
    gives the fastest results.
    """
    # cdef object fcntype
    # cdef public object fcn
    def __cinit__(self, *args, **kargs):
        self.fcntype = 'batch'
        self.fcn = None

    # def _get_rank(self):
    #     return 0 if mpi4py is None else mpi4py.MPI.COMM_WORLD.Get_rank()

    # rank = property(
    #     _get_rank, doc='MPI rank (deprecated - use Integrator.mpi_rank)'
    #     )

    def __call__(self, x):
        try:
            return self.fcn(x)
        except TypeError:
            raise TypeError('no __call__ method defined (or badly defined)')

def batchintegrand(f):
    """ Decorator for batch integrand functions.

    Applying :func:`vegas.batchintegrand` to a function ``fcn`` repackages
    the function in a format that |vegas| can understand. Appropriate
    functions take a :mod:`numpy` array of integration points ``x[i, d]``
    as an argument, where ``i=0...`` labels the integration point and
    ``d=0...`` labels direction, and return an array ``f[i]`` of
    integrand values (or arrays of integrand values) for the corresponding
    points. The meaning of ``fcn(x)`` is unchanged by the decorator.

    An example is ::

        import vegas
        import numpy as np

        @vegas.batchintegrand
        def f(x):
            return np.exp(-x[:, 0] - x[:, 1])

    for the two-dimensional integrand :math:`\exp(-x_0 - x_1)`.

    This decorator provides an alternative to deriving an integrand
    class from :class:`vegas.BatchIntegrand`.
    """
    try:
        f.fcntype = 'batch'
        # f.rank = 0 if mpi4py is None else mpi4py.MPI.COMM_WORLD.Get_rank()
        return f
    except:
        ans = BatchIntegrand()
        ans.fcn = f
        return ans

# legacy names
vecintegrand = batchintegrand
MPIintegrand = batchintegrand



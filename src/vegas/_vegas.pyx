# c#ython: profile=True

# Created by G. Peter Lepage (Cornell University) in 12/2013.
# Copyright (c) 2013 G. Peter Lepage. 
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
from libc.math cimport floor, log, abs, tanh, erf, exp, sqrt

import sys
import numpy 
import math

cdef double TINY = 1e-308                            # smallest and biggest
cdef double HUGE = 1e308


have_gvar = False
try:
    import gvar
    have_gvar = True
except ImportError:
    have_scipy = False
    try:
        import scipy.special
        have_scipy = True
    except ImportError:
        pass
    # fake version of gvar.gvar
    # for use if lsqfit module not available
    class GVar(object):
        def __init__(self, mean, sdev):
            self.mean = float(mean)
            self.sdev = float(sdev)
            self.internaldata = (self.mean, self.sdev)
        def __str__(self):
            return "%g +- %g" % (self.mean, self.sdev)
    class _gvar_standin:
        def __init__(self):
            pass
        def gvar(self, mean, sdev):
            return GVar(mean, sdev)
        def gammaQ(self, a, b):
            if have_scipy:
                return scipy.special.gammaincc(a, b)
            else:
                return -1.
        def mean(self, glist):
            return numpy.array([g.mean for g in glist])
        def var(self, glist):
            return numpy.array([g.sdev ** 2 for g in glist])

    gvar = _gvar_standin()
    gvar.GVar = GVar

class RunningWAvg(gvar.GVar):
    """ Running weighted average of Monte Carlo estimates.

    This class accumulates independent Monte Carlo 
    estimates (e.g., of an integral) and combines 
    them into a single weighted average. It 
    is derived from :class:`gvar.GVar` (from 
    the :mod:`lsqfit` module if it is present) and 
    represents a Gaussian random variable.
    """
    def __init__(self, gvar_list=None):
        if gvar_list is not None:
            self.__init__()
            for g in gvar_list:
                self.add(g)
        else:
            self._v_s2 = 0.
            self._v2_s2 = 0.
            self._1_s2 = 0.
            self.itn_results = []
            super(RunningWAvg, self).__init__(
                *gvar.gvar(0., 0.).internaldata,
               )

    def _chi2(self):
        return self._v2_s2 - self._v_s2 ** 2 / self._1_s2
    chi2 = property(_chi2, None, None, "*chi**2* of weighted average.")

    def _dof(self):
        return len(self.itn_results) - 1
    dof = property(
        _dof, 
        None, 
        None, 
        "Number of degrees of freedom in weighted average."
        )

    def _Q(self):
        return (
            gvar.gammaQ(self.dof / 2., self.chi2 / 2.)
            if self.dof > 0
            else 1
            )
    Q = property(
        _Q, 
        None, 
        None, 
        "*Q* or *p-value* of weighted average's *chi**2*.",
        )

    def add(self, g):
        """ Add estimate ``g`` to the running average. """
        self.itn_results.append(g)
        var = g.sdev ** 2
        assert var > 0.0, 'zero variance not allowed'          
        self._v_s2 = self._v_s2 + g.mean / var
        self._v2_s2 = self._v2_s2 + g.mean ** 2 / var
        self._1_s2 = self._1_s2 + 1. / var
        super(RunningWAvg, self).__init__(*gvar.gvar(
            self._v_s2 / self._1_s2,
            sqrt(1. / self._1_s2),
            ).internaldata)
    def summary(self):
        """ Assemble summary of independent results into a string. """
        acc = RunningWAvg()
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
        ans = fmt % ('itn', 'integral', 'wgt average', 'chi2/dof', 'Q')
        ans += len(ans[:-1]) * '-' + '\n'
        for data in linedata:
            ans += fmt % data
        return ans

# AdaptiveMap is used by Integrator 
cdef class AdaptiveMap:
    """ Adaptive map ``y->x(y)`` for multidimensional ``y`` and ``x``.

    An :class:`AdaptiveMap` defines a multidimensional map ``y -> x(y)`` 
    from the unit hypercube, with ``0 <= y[d] <= 1``, to an arbitrary
    hypercube in ``x`` space. Each direction is mapped independently 
    with a jacobian that is tunable (i.e., "adaptive").

    The map is specified by a grid in ``x``-space that, by definition, 
    maps into a uniformly space grid in ``y``-space. The nodes of 
    the grid are specified by ``grid[d, i]`` where d is the 
    direction (``d=0,1...dim-1``) and ``i`` labels the grid point
    (``i=0,1...N-1``). The mapping for specific point ``y`` into
    ``x`` space is:: 

        y[d] -> x[d] = grid[d, i(y[d])] + inc[d, i(y[d])] * delta(y[d])

    where ``i(y)=floor(y*N``), ``delta(y)=y*N-i(y)``, and
    ``inc[d, i] = grid[d, i+1] - grid[d, i]``. The jacobian for this map, :: 

        dx[d]/dy[d] = inc[d, i(y[d])] * N,

    is piece-wise constant and proportional to the ``x``-space grid 
    spacing. Each increment in the ``x``-space grid maps into an increment of 
    size ``1/N`` in the corresponding ``y`` space. So increments with 
    small ``delta_x[i]`` are stretched out in ``y`` space, while larger
    increments are shrunk.

    The ``x`` grid for an :class:`AdaptiveMap` can be specified explicitly
    when it is created: for example, ::

        map = AdaptiveMap([[0, 0.1, 1], [-1, 0, 1]])

    creates a two-dimensional map where the ``x[0]`` interval ``(0,0.1)``
    and ``(0.1,1)`` map into the ``y[0]`` intervals ``(0,0.5)`` and 
    ``(0.5,1)`` respectively, while ``x[1]`` intervals ``(-1.,0)`` 
    and ``(0,1)`` map into ``y[1]`` intervals ``(0,0.5)`` and  ``(0.5,1)``.

    More typically an initially uniform map is trained so that 
    ``F(x(y), dx(y)/dy)``, for some training function ``F``, 
    is (approximately) constant across ``y`` space. The training function
    is assumed to grow monotonically with the jacobian ``dx(y)/dy`` at
    fixed ``x``. The adaptation is done iteratively, beginning 
    with a uniform map::

        map = AdaptiveMap([[xl[0], xu[0]], [xl[1], xu[1]]...], ninc=N)

    which creates an ``x`` grid with ``N`` equal-sized increments 
    between ``x[d]=xl[d]`` and ``x[d]=xu[d]``. The training function 
    is then evaluated for the ``x`` values corresponding to 
    a list of ``y`` values ``y[i, d]`` spread over ``(0,1)`` for 
    each direction ``d``::

        ...
        for i in range(ny):
            for d in range(dim):
                y[i, d] = ....
        x = numpy.empty(y.shape, float)     # container for corresponding x's
        jac = numpy.empty(y.shape, float)   # container for corresponding dx/dy's
        map.map(y, x, jac)                  # fill x and jac
        f = F(x, jac)                       # compute training function

    The number of ``y`` points is arbitrary, but typically large. Training 
    data is often generated in batches that are accumulated by the 
    map through multiple calls to::

        map.add_training_data(y, f)

    Finally the map is adapted to the data::

        m.adapt(alpha=1.5)

    The process of computing training data and then adapting the map
    typically has to be repeated several times before the map converges,
    at which point the ``x``-grid's nodes, ``map.grid[d, i]``, stop changing.

    The speed with which the grid adapts is determined by parameter ``alpha``.
    Large (positive) values imply rapid adaptation, while small values (much
    less than one) imply slow adaptation. As in any iterative process, it is
    usually a good idea to slow adaptation down in order to avoid
    instabilities.

    :param grid: Initial ``x`` grid, where ``grid[d, i]`` is the ``i``-th 
        node in direction ``d``.
    :type x: 2-d array of floats
    :param ninc: Number of increments along each axis of the ``x`` grid. 
        A new grid is generated if ``ninc`` differs from ``grid.shape[1]``.
        The new grid is designed to give the same jacobian ``dx(y)/dy``
        as the original grid. The default value, ``ninc=None``,  leaves 
        the grid unchanged.
    :type ninc: ``int`` or ``None``
    """
    def __init__(self, grid, ninc=None):
        cdef INT_TYPE i, d
        if isinstance(grid, AdaptiveMap):
            self._init_grid(grid.grid, initinc=True)
        else:
            grid = numpy.array(grid, float)
            if grid.ndim != 2:
                raise ValueError('grid must be 2-d array not %d-d' % grid.ndim)
            grid.sort(axis=1)
            if grid.shape[1] < 2: 
                raise ValueError("grid.shape[1] smaller than 2: " % grid.shape[1])
            self._init_grid(grid, initinc=True)
        self.sum_f = None
        self.n_f = None
        if ninc is not None and ninc != self.inc.shape[1]:
            if self.inc.shape[1] == 1:
                self.make_uniform(ninc=ninc)
            else:
                self.adapt(ninc=ninc)

    property ninc:
        " Number of increments along each grid axis."
        def __get__(self):
            return self.inc.shape[1] 
    property  dim:
        " Number of dimensions."
        def __get__(self):
            return self.grid.shape[0]
    def region(self, INT_TYPE d=-1):
        """ x-space region.

        ``region(d)`` returns a tuple ``(xl,xu)`` specifying the ``x``-space
        interval covered by the map in direction ``d``. A list containing
        the intervals for each direction is returned if ``d`` is omitted.
        """
        if d < 0:
            return [self.region(d) for d in range(self.dim)]
        else:
            return (self.grid[d, 0], self.grid[d, -1])

    def __reduce__(self):
        """ Capture state for pickling. """
        return (AdaptiveMap, (numpy.asarray(self.grid),))

    def settings(self, ngrid=5):
        """ Create string with information about grid nodes.

        Creates a string containing the locations of the nodes
        in the map grid for each direction. Parameter 
        ``ngrid`` specifies the maximum number of nodes to print
        (spread evenly over the grid).
        """
        ans = []
        if ngrid > 0:
            grid = numpy.array(self.grid)
            nskip = int(self.ninc // ngrid)
            if nskip<1:
                nskip = 1
            start = nskip // 2
            for d in range(self.dim):
                ans += [
                    "    grid[%2d] = %s" 
                    % (
                        d, 
                        numpy.array2string(
                            grid[d, start::nskip],precision=3,
                            prefix='    grid[xx] = ')
                          )
                    ]
        return '\n'.join(ans) + '\n'

    def make_uniform(self, ninc=None):
        """ Replace the grid with a uniform grid.

        The new grid has ``ninc`` increments along each direction if 
        ``ninc`` is specified. Otherwise it has the same number of 
        increments as the old grid.
        """
        cdef INT_TYPE i, d
        cdef INT_TYPE dim = self.grid.shape[0]
        cdef double[:] tmp
        cdef double[:, ::1] new_grid
        if ninc is None:
            ninc = self.inc.shape[1]
        ninc = int(ninc)
        if ninc < 1:
            raise ValueError(
                "no of increments < 1 in AdaptiveMap -- %d" 
                % ninc
                )
        new_grid = numpy.empty((dim, ninc + 1), float)
        for d in range(dim):
            tmp = numpy.linspace(self.grid[d, 0], self.grid[d, -1], ninc + 1)
            for i in range(ninc + 1):
                new_grid[d, i] = tmp[i]
        self._init_grid(new_grid)

    def _init_grid(self, new_grid, initinc=False):
        " Set the grid equal to new_grid. "
        cdef INT_TYPE dim = new_grid.shape[0]
        cdef INT_TYPE ninc = new_grid.shape[1] - 1 
        cdef INT_TYPE d, i
        self.grid = new_grid
        if initinc or self.inc.shape[0] != dim or self.inc.shape[1] != ninc:
            self.inc = numpy.empty((dim, ninc), float)
        for d in range(dim):
            for i in range(ninc):
                self.inc[d, i] = self.grid[d, i + 1] - self.grid[d, i]

    def __call__(self, y):
        """ Return ``x`` values corresponding to ``y``. 

        ``y`` can be a single ``dim``-dimensional point, or it 
        can be an array ``y[i,j, ..., d]`` of such points (``d=0..dim-1``).
        """
        y = numpy.asarray(y, float)
        y_shape = y.shape
        y.shape = -1, y.shape[-1]
        x = 0 * y
        jac = numpy.empty(y.shape[0], float)
        self.map(y, x, jac)
        x.shape = y_shape
        return x

    def jac(self, y):
        """ Return the map's jacobian at ``y``. 

        ``y`` can be a single ``dim``-dimensional point, or it 
        can be an array ``y[d,i,j,...]`` of such points (``d=0..dim-1``).
        """
        y = numpy.asarray(y)
        y_shape = y.shape
        y.shape = -1, y.shape[-1]
        x = 0 * y
        jac = numpy.empty(y.shape[0], float)
        self.map(y, x, jac)
        return jac

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    cpdef map(
        self, 
        double[:, ::1] y, 
        double[:, ::1] x, 
        double[::1] jac, 
        INT_TYPE ny=-1
        ):
        """ Map y to x, where jac is the jacobian.

        ``y[i, d]`` is an array of ``ny`` ``y``-values for direction ``d``.
        ``x[i, d]`` is filled with the corresponding ``x`` values,
        and ``jac[i]`` is filled with the corresponding jacobian 
        values. ``x`` and ``jac`` must be preallocated: for example, ::

            x = numpy.empty(y.shape, float)
            jac = numpy.empty(y.shape[0], float)

        :param y: ``y`` values to be mapped. ``y`` is a contiguous 2-d array,
            where ``y[i, d]`` contains values for points along direction ``d``.
        :type y: contiguous 2-d array of floats
        :param x: Container for ``x`` values corresponding to ``y``.
        :type x: contiguous 2-d array of floats
        :param jac: Container for jacobian values corresponding to ``y``.
        :type jac: contiguous 1-d array of floats
        :param ny: Number of ``y`` points: ``y[i, d]`` for ``d=0...dim-1``
            and ``i=0...ny-1``. ``ny`` is set to ``y.shape[0]`` if it is
            omitted (or negative).
        :type ny: int
        """
        cdef INT_TYPE ninc = self.inc.shape[1]
        cdef INT_TYPE dim = self.inc.shape[0]
        cdef INT_TYPE i, iy, d  
        cdef double y_ninc, dy_ninc, tmp_jac
        if ny < 0:
            ny = y.shape[0]
        elif ny > y.shape[0]:
            raise ValueError('ny > y.shape[0]: %d > %d' % (ny, y.shape[0]))
        for i in range(ny):
            jac[i] = 1.
            for d in range(dim):
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

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    cpdef add_training_data(
        self, 
        double[:, ::1] y, 
        double[::1] f, 
        INT_TYPE ny=-1,
        ):
        """ Add training data ``f`` for ``y``-space points ``y``.

        Accumulates training data for later use by ``self.adapt()``.

        :param y: ``y`` values corresponding to the training data. 
            ``y`` is a contiguous 2-d array, where ``y[i, d]`` 
            is for points along direction ``d``.
        :type y: contiguous 2-d array of floats
        :param f: Training function values. ``f[i]`` corresponds to 
            point ``y[i, d]`` in ``y``-space.
        :type f: contiguous 2-d array of floats
        :param ny: Number of ``y`` points: ``y[i, d]`` for ``d=0...dim-1``
            and ``i=0...ny-1``. ``ny`` is set to ``y.shape[0]`` if it is
            omitted (or negative).
        :type ny: int
        """
        cdef INT_TYPE ninc = self.inc.shape[1]
        cdef INT_TYPE dim = self.inc.shape[0]
        cdef INT_TYPE iy 
        cdef INT_TYPE i, d
        if self.sum_f is None:
            self.sum_f = numpy.zeros((dim, ninc), float)
            self.n_f = numpy.zeros((dim, ninc), float) + TINY
        if ny < 0:
            ny = y.shape[0]
        elif ny > y.shape[0]:
            raise ValueError('ny > y.shape[0]: %d > %d' % (ny, y.shape[0]))
        for d in range(dim):
            for i in range(ny):
                iy = <int> floor(y[i, d] * ninc)
                self.sum_f[d, iy] += abs(f[i])
                self.n_f[d, iy] += 1
        return
        
    def adapt(self, double alpha=0.0, ninc=None):
        """ Adapt grid to accumulated training data.

        The new grid is designed to make the training function constant
        in ``y[d]`` when the ``y``\s in the other directions  are integrated out.
        The number of increments along a direction can be changed 
        by setting parameter ``ninc``. 

        The grid does not change if no training data has been accumulated,
        unless ``ninc`` is specified, in which case the number of 
        increments is adjusted while preserving the relative density
        of increments at different values of ``x``.

        :parameter alpha: Determines the speed with which the grid adapts to 
            training data. Large (postive) values imply rapid evolution; 
            small values (much less than one) imply slow evolution. Typical 
            values are of order one. Choosing ``alpha<0`` causes adaptation
            to the unmodified training data (usually not a good idea).
        :type alpha: double or None
        :parameter ninc: Number of increments along each direction in the 
            new grid. The number is unchanged from the old grid if ``ninc``
            is omitted (or equals ``None``).
        :type ninc: int or None
        """
        cdef double[:, ::1] new_grid
        cdef double[::1] avg_f, tmp_f
        cdef double sum_f, acc_f, f_inc
        cdef INT_TYPE old_ninc = self.grid.shape[1] - 1
        cdef INT_TYPE dim = self.grid.shape[0]
        cdef INT_TYPE i, j, new_ninc
        cdef double smth = 3.   # was 3.
        #
        # initialization
        if ninc is None:
            new_ninc = old_ninc
        else:
            new_ninc = ninc
        if new_ninc < 1:
            raise ValueError('ninc < 1: ' + str(new_ninc))
        if new_ninc == 1:
            new_grid = numpy.empty((dim, 2), float)
            for d in range(dim):
                new_grid[d, 0] = self.grid[d, 0]
                new_grid[d, 1] = self.grid[d, -1]
            self._init_grid(new_grid)
            return
        #
        # smoothing
        new_grid = numpy.empty((dim, new_ninc + 1), float)
        avg_f = numpy.ones(old_ninc, float) # default = uniform
        for d in range(dim):
            if self.sum_f is not None and alpha != 0:
                for i in range(old_ninc):
                    avg_f[i] = self.sum_f[d, i] / self.n_f[d, i]
            if alpha > 0 and old_ninc > 1:
                tmp_f = numpy.empty(old_ninc, float)
                tmp_f[0] = (3. * avg_f[0] + avg_f[1]) / 4.
                tmp_f[-1] = (3. * avg_f[-1] + avg_f[-2]) / 4.
                sum_f = tmp_f[0] + tmp_f[-1]
                for i in range(1, old_ninc - 1):
                    tmp_f[i] = (6. * avg_f[i] + avg_f[i-1] + avg_f[i+1]) / 8.
                    sum_f += tmp_f[i]
                if sum_f > 0:
                    for i in range(old_ninc):
                        avg_f[i] = tmp_f[i] / sum_f + TINY
                else:
                    for i in range(old_ninc):
                        avg_f[i] = TINY
                for i in range(old_ninc):
                    avg_f[i] = (-(1 - avg_f[i]) / log(avg_f[i])) ** alpha
            #
            # regrid
            new_grid[d, 0] = self.grid[d, 0]
            new_grid[d, -1] = self.grid[d, -1]
            i = 0        # new_x index
            j = -1         # self_x index
            acc_f = 0   # sum(avg_f) accumulated
            f_ninc = 0.
            for i in range(old_ninc):
                f_ninc += avg_f[i]
            f_ninc /= new_ninc     # amount of acc_f per new increment
            for i in range(1, new_ninc):
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
        self._init_grid(new_grid)
        self.sum_f = None
        self.n_f = None

    def plot_grid(self, ngrid=40, shrink=False):
        """ Display plots showing the current grid. 

        :param ngrid: The number of grid nodes in each 
            direction to include in the plot. The default is 40.
        :type ngrid: int 
        :param shrink: Displays entire range of each maxinc_axis
            if ``False``; otherwise shrink range to include
            just the nodes being displayed. The default is
            ``False``. 
        """
        import matplotlib.pyplot as plt 
        fig = plt.figure()
        nskip = int(self.ninc // ngrid)
        if nskip < 1:
            nskip = 1
        start = nskip // 2
        def plotdata(idx, grid=numpy.asarray(self.grid)):
            if idx[0] >= grid.shape[0]:
                idx[0] -= 1
            elif idx[0] < 0:
                idx[0] = 0
            dx = idx[0]
            dy = dx + 1
            xrange = [self.grid[dx, 0], self.grid[dx, -1]]
            xgrid = grid[dx, start::nskip]
            xlabel = 'x[%d]' % dx 
            if dy < grid.shape[0]:
                yrange = [self.grid[dy, 0], self.grid[dy, -1]]
                ygrid = grid[dy, start::nskip]
                ylabel = 'x[%d]' % dy
                fig_caption = 'axes %d, %d' % (dx, dy)
            else:
                yrange = [0., 1.]
                ygrid = None
                ylabel = ''
                fig_caption = 'axis %d' % dx
            if shrink:
                xrange = [min(xgrid), max(xgrid)]
                if ygrid is not None:
                    yrange = [min(ygrid), max(ygrid)]
            fig.clear()
            plt.title(
                "%s   (press 'n', 'p', 'q' or a digit)"
                % fig_caption
                )
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            # ax = fig.add_subplot(111)
            # ax.set_ylabel(ylabel)
            # ax.set_xlabel(xlabel)
            for i in range(len(xgrid)):
                plt.plot([xgrid[i], xgrid[i]], yrange, 'k-')
                if ygrid is not None:
                    plt.plot(xrange, [ygrid[i], ygrid[i]], 'k-')
            if shrink:
                plt.xlim(*xrange)
                if ygrid is not None:
                    plt.ylim(*yrange)
            elif ygrid is None:
                plt.xlim(*xrange)
            plt.draw()

        idx = [0]        
        def onpress(event, idx=idx):
            try:    # digit?
                idx[0] = int(event.key)
            except ValueError:
                if event.key == 'n':
                    idx[0] += 1
                elif event.key == 'p':
                    idx[0] -= 1
                elif event.key == 'q':
                    plt.close()
                    return
                else:
                    return
            plotdata(idx)
        
        fig.canvas.mpl_connect('key_press_event', onpress)
        plotdata(idx)
        plt.show()       

cdef class Integrator(object):
    """ Adaptive multidimensional Monte Carlo integration.

    :class:`vegas.Integrator` objects make Monte Carlo 
    estimates of multidimensional functions ``f(x)``
    where ``x[d]`` is a point in the integration volume::

        integ = vegas.Integrator(integration_region)

        result = integ(f, nitn=10, neval=10000)

    The integator makes ``nitn`` estimates of the integral, 
    each using
    at most ``neval`` samples of the integrand,
    as it adapts to the specific features
    of the integrand. Successive estimates typically improve
    in accuracy until the integrator has fully adapted.
    The integrator returns the weighted average of all ``nitn``
    estimates, together with an estimate of the statistical
    (Monte Carlo) uncertainty in that estimate of the integral. The 
    result is an object of type :class:`RunningWAvg`
    (which is derived from :class:`gvar.GVar`).

    :param map: The integration region as specified by 
        an array ``xlimit[d, i]`` where ``d`` is the 
        direction and ``i=0,1`` specify the lower
        and upper limits of integration in direction ``d``.

        ``map`` could also be the integration map from 
        another :class:Integrator, or the |Integrator|
        itself. In this case the grid is copied from the 
        existing integrator.
    :type map: array or :class:`vegas.AdaptiveMap` or :class:`vegas.Integrator`
    :param nitn: The maximum number of iterations used to 
        adapt to the integrand and estimate its value. The
        default value is 10.
    :type nitn: positive int
    :param neval: The maximum number of integrand evaluations
        in each iteration of the |vegas| algorithm. Default 
        value is 1000.
    :type neval: positive int 
    :param alpha: Damping parameter controlling the remapping
        of the integration variables as |vegas| adapts to the
        integrand. Smaller values slow adaptation, which may 
        be desirable for difficult integrands. The default value 
        is 0.5.
    :type alpha: float 
    :param beta: Damping parameter controlling the redistribution
        of integrand evaluations across hypercubes in the 
        stratified sampling of the integrand (over transformed
        variables). Smaller values limit the amount of 
        redistribution. The theoretically optimal value is 1;
        setting ``beta=0`` prevents any redistribution of 
        evaluations. The default value is 0.75.
    :type beta: float 
    :param nhcube_vec: The number of hypercubes (in |y| space)
        whose integration points are combined into a single
        vector to be passed to the integrand 
        when using |vegas| in vectorized mode. The default
        value is 100.
    :type nhcube_vec: positive int 
    :param maxinc_axis: The maximum number of increments
        per axis allowed for the |x|-space grid. The default 
        value is 1000. 
    :type maxinc_axis: positive int 
    :param mode: ``mode=adapt_to_integrand`` causes 
        |vegas| to remap the integration variables to emphasize
        regions where ``|f(x)|`` is largest. This is 
        the default mode.

        ``mode=adapt_to_errors`` causes |vegas| to remap 
        variables to emphasize regions where the Monte Carlo
        error is largest. This might be superior when 
        the number of the number of stratifications in 
        the |y| grid is large (> 50?). It is typically 
        useful only in one or two dimensions.
    :param fcntype: Specifies the default type of integrand.

        ``fcntype='scalar'`` imples that the integrand
        is a function ``f(x)`` of a single integration
        point. 

        ``fcntype='vector'`` implies that 
        the integrand function takes three arguments:
        a list multiple integration points ``x[i, d]``,
        where ``i=0...nx-1`` labels the integration point
        and ``d`` labels the direction; a buffer
        ``f[i]`` into which the corresponding 
        integrand values are written; and the number
        ``nx`` of integration points provided. 

        The 
        default is ``fcntype=scalar``, but this is 
        overridden if the integrand has a ``fcntype`` 
        attribute. It is also overridden for classes
        derived from :class:`VecIntegrand`, which are
        treated as ``fcntype='vector'`` integrands.
    :param rtol: Relative error in the integral estimate 
        at which point the integrator can stop. The default
        value is 0.0 which means that the integrator will
        complete all iterations specified by ``nitn``.
    :type rtol: float less than 1
    :param atol: Absolute error in the integral estimate 
        at which point the integrator can stop. The default
        value is 0.0 which means that the integrator will
        complete all iterations specified by ``nitn``.
    :type atol: float 
    :param analyzer: An object with methods 

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
        of its results as they are produced.
    """

    # Settings accessible via the constructor and Integrator.set
    defaults = dict(
        map=None,
        fcntype='scalar', # default integrand type
        neval=1000,       # number of evaluations per iteration
        maxinc_axis=1000,  # number of adaptive-map increments per axis
        nhcube_vec=100,    # number of h-cubes per vector
        nitn=10,           # number of iterations
        alpha=0.5,
        beta=0.75,
        mode='adapt_to_integrand',
        rtol=0,
        atol=0,
        analyzer=None,
        )

    def __init__(self, map, **kargs):
        # N.B. All attributes initialized automatically by cython.
        #      This is why self.set() works here.
        if isinstance(map, Integrator):
            args = {}
            for k in Integrator.defaults:
                args[k] = getattr(map, k)
            args.update(kargs)
            self.set(args)
        else:
            args = dict(Integrator.defaults)
            del args['map']
            args.update(kargs)
            self.set(args)
            self.map = AdaptiveMap(map)
        self.sigf_list = numpy.array([], float) # dummy
        self.neval_hcube_range = None
        self.last_neval = 0

    def __reduce__(self):
        """ Capture state for pickling. """
        odict = dict()
        for k in Integrator.defaults:
            if k in ['map']:
                continue
            odict[k] = getattr(self, k)
        return (Integrator, (self.map,), odict)

    def __setstate__(self, odict):
        """ Set state for unpickling. """
        for k in odict:
            setattr(self, k, odict[k])

    def set(self, ka={}, **kargs):
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
        if kargs:
            kargs.update(ka)
        else:
            kargs = ka
        old_val = dict() 
        for k in kargs:
            if k == 'map':
                old_val[k] = self.map
                self.map = AdaptiveMap(kargs[k])
            elif k == 'fcntype':
                old_val[k] = self.fcntype
                self.fcntype = kargs[k]
            elif k == 'neval':
                old_val[k] = self.neval
                self.neval = kargs[k]
            elif k == 'maxinc_axis':
                old_val[k] = self.maxinc_axis
                self.maxinc_axis = kargs[k]
            elif k == 'nhcube_vec':
                old_val[k] = self.nhcube_vec
                self.nhcube_vec = kargs[k]
            elif k == 'nitn':
                old_val[k] = self.nitn
                self.nitn = kargs[k]
            elif k == 'alpha':
                old_val[k] = self.alpha
                self.alpha = kargs[k]
            elif k == 'mode':
                old_val[k] = self.mode
                self.mode = kargs[k]
            elif k == 'beta':
                old_val[k] = self.beta
                self.beta = kargs[k]
            elif k == 'analyzer':
                old_val[k] = self.analyzer
                self.analyzer = kargs[k]
            elif k == 'rtol':
                old_val[k] = self.rtol
                self.rtol = kargs[k]
            elif k == 'atol':
                old_val[k] = self.atol
                self.atol = kargs[k]
            else:
                raise AttributeError('no attribute named "%s"' % str(k))
        return old_val

    def settings(self, ngrid=0):
        """ Assemble summary of integrator settings into string.

        :param ngrid: Number of grid nodes in each direction 
            to include in summary.
            The default is 0.
        :type ngrid: int
        :returns: String containing the settings.
        """
        cdef INT_TYPE d
        mode = self.mode
        beta = self.beta # if mode != 'adapt_to_errors' else 0.0
        nhcube = self.nstrat ** self.dim
        neval = nhcube * self.neval_hcube if self.beta <= 0 else self.neval
        ans = ""
        ans = "Integrator Settings:\n"
        if beta > 0:
            ans = ans + (
                "    %d (max) integrand evaluations in each of %d iterations\n"
                % (self.neval, self.nitn)
                )
        else:
            ans = ans + (
                "    %d integrand evaluations in each of %d iterations\n"
                % (neval, self.nitn)
                )
        ans = ans + ("    integrator mode = %s\n" % mode)
        ans = ans + (
            "    number of:  strata/axis = %d  increments/axis = %d\n"
            % (self.nstrat, self.map.ninc)
            )
        if beta > 0:
            ans = ans + (
                "                h-cubes = %d  evaluations/h-cube = %d (min)\n"
                % (nhcube, self.neval_hcube)
                )
        else:
            ans = ans + (
                    "                h-cubes = %d  evaluations/h-cube = %d\n"
                    % (nhcube, self.neval_hcube)
                    )
        ans += "                h-cubes/vector = %d\n" % self.nhcube_vec
        ans = ans + (
            "    damping parameters: alpha = %g  beta= %g\n" 
            % (self.alpha, beta)
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

    def _prepare_integrator(self):
        """ Prep the integrator before integrating.

        Decide how many increments and strata to use
        and allocate space to store the sigf values.
        """
        # determine # of strata, # of increments
        dim = self.map.dim
        neval_eff = (self.neval / 2.0) if self.beta > 0 else self.neval
        ns = int((neval_eff / 2.) ** (1. / dim))# stratifications/axis
        ni = int(self.neval / 10.)              # increments/axis
        if ns < 1:
            ns = 1
        if ni < 1:
            ni = 1
        elif ni  > self.maxinc_axis:
            ni = self.maxinc_axis
        # want even number increments in each stratification 
        # or vise versa
        if ns > ni:
            if ns < self.maxinc_axis:
                ni = ns
            else:
                ns = int(ns // ni) * ni
        else:
            ni = int(ni // ns) * ns
        if self.mode == 'adapt_to_errors':
            # ni > ns makes no sense with this mode
            if ni > ns:
                ni = ns
            # beta > 0 incompatible with this mode
            # self.beta = 0.

        self.map.adapt(ninc=ni)    

        self.nstrat = ns
        nhcube = self.nstrat ** dim
        self.neval_hcube = int(floor(neval_eff // nhcube))
        if self.neval_hcube < 2:
            self.neval_hcube = 2
        self.dim = dim
        if self.beta > 0 and len(self.sigf_list) != nhcube:
            self.sigf_list = numpy.ones(nhcube, float)

    def _prepare_integrand(self, fcn, fcntype=None):
        """ Wrap integrand if needed. """
        if fcntype is None:
            fcntype = self.fcntype
        if hasattr(fcn, 'fcntype'):
            fcntype = fcn.fcntype
        if fcntype == 'scalar':
            return VecPythonIntegrand(fcn)
        else:
            return fcn

    def __call__(self, fcn, **kargs):
        """ Integrate ``fcn``, possibly overriding default parameters. """
        cdef INT_TYPE itn
        # determine fcntype from fcn, self or kargs 
        if 'fcntype' in kargs:
            fcntype = kargs['fcntype']
            del kargs['fcntype']
        else:
            fcntype = self.fcntype
        fcn = self._prepare_integrand(fcn, fcntype=fcntype)

        # save old settings so they can be restored at end
        if kargs:
            old_kargs = self.set(kargs)
        else:
            old_kargs = {}

        # main iteration loop
        self._prepare_integrator()
        ans = RunningWAvg()
        for itn in range(self.nitn):    # iterate
            if self.analyzer != None:
                self.analyzer.begin(itn, self)
            ans.add(self._integrate(fcn))
            if self.analyzer != None:
                self.analyzer.end(ans.itn_results[-1], ans)
            self.map.adapt(alpha=self.alpha)
            if (self.rtol * abs(ans.mean) + self.atol) > ans.sdev:
                break

        # do not restore old settings
        # if old_kargs:
        #     self.set(old_kargs) 

        return ans

    def debug(self, fcn, INT_TYPE neval=10):
        """ Evaluate fcn at ``neval`` random points. """
        cdef INT_TYPE i
        cdef double[:, ::1] y = numpy.random.uniform(
            0., 1., (self.map.dim, neval)
            )
        cdef double[:, ::1] x = self.map(y)
        cdef double[::1] f = numpy.empty(neval, float)
        fcn = self._prepare_integrand(fcn)
        fcn(x, f, neval)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _integrate(self, fcn):
        """ Do integral for one iteration. """
        cdef INT_TYPE nhcube = self.nstrat ** self.dim 
        cdef INT_TYPE nhcube_vec = min(self.nhcube_vec, nhcube)
        cdef INT_TYPE adapt_to_integrand = (
            1 if self.mode == 'adapt_to_integrand' else 0
            )
        cdef INT_TYPE redistribute = (
            # 1 if (self.beta > 0 and adapt_to_integrand) else 0
            1 if (self.beta > 0) else 0
            )
        cdef double neval_sigf = (
            self.neval / 2. / numpy.sum(self.sigf_list)
            ) if redistribute else 0
        cdef INT_TYPE[::1] y0 = numpy.empty(self.dim, int)
        cdef double dv = (1./self.nstrat) ** self.dim
        cdef double ans_mean = 0.
        cdef double ans_var = 0.
        cdef INT_TYPE neval_vec = self.nhcube_vec * self.neval_hcube
        cdef double[:, ::1] y = numpy.empty((neval_vec, self.dim), float)
        cdef double[:, ::1] x = numpy.empty((neval_vec, self.dim), float)
        cdef double[::1] jac = numpy.empty(neval_vec, float)
        cdef INT_TYPE min_neval_hcube = self.neval_hcube
        cdef INT_TYPE max_neval_hcube = self.neval_hcube
        cdef INT_TYPE hcube, i, j, d, hcube_base, ihcube, i_start
        cdef INT_TYPE[::1] neval_hcube = (
            numpy.zeros(self.nhcube_vec, int) + self.neval_hcube 
            )
        cdef double sum_fdv
        cdef double sum_fdv2
        cdef double[::1] fdv = numpy.empty(neval_vec, float)
        cdef double[::1] fdv2 = numpy.empty(neval_vec, float)
        cdef double sigf2
        cdef double[:, ::1] yran
        cdef INT_TYPE tmp_hcube, counter = 0 ########

        # iterate over h-cubes in batches of nhcube_vec h-cubes
        # this allows for vectorization, to reduce python overhead
        self.last_neval = 0
        nhcube_vec = self.nhcube_vec
        for hcube_base in range(0, nhcube, nhcube_vec):
            if (hcube_base + nhcube_vec) > nhcube:
                nhcube_vec = nhcube - hcube_base 

            # compute neval_hcube for each h-cube
            # reinitialize work areas if necessary
            if redistribute:
                neval_vec = 0
                for ihcube in range(nhcube_vec):
                    neval_hcube[ihcube] = <int> (
                        self.sigf_list[hcube_base + ihcube] * neval_sigf
                        )
                    if neval_hcube[ihcube] < self.neval_hcube:
                        # counter += 1
                        neval_hcube[ihcube] = self.neval_hcube
                    if neval_hcube[ihcube] < min_neval_hcube:
                        min_neval_hcube = neval_hcube[ihcube]
                    if neval_hcube[ihcube] > max_neval_hcube:
                        max_neval_hcube = neval_hcube[ihcube]
                    neval_vec += neval_hcube[ihcube]
                if neval_vec > y.shape[0]:
                    y = numpy.empty((neval_vec, self.dim), float)
                    x = numpy.empty((neval_vec, self.dim), float)
                    jac = numpy.empty(neval_vec, float)    
                    fdv = numpy.empty(neval_vec, float)  
                    fdv2 = numpy.empty(neval_vec, float) 
            self.last_neval += neval_vec
           
            i_start = 0
            yran = numpy.random.uniform(0., 1., (neval_vec, self.dim))
            for ihcube in range(nhcube_vec):
                hcube = hcube_base + ihcube
                tmp_hcube = hcube
                for d in range(self.dim):
                    y0[d] = tmp_hcube % self.nstrat
                    tmp_hcube = (tmp_hcube - y0[d]) / self.nstrat
                for d in range(self.dim):
                    for i in range(i_start, i_start + neval_hcube[ihcube]):
                        y[i, d] = (y0[d] + yran[i, d]) / self.nstrat
                i_start += neval_hcube[ihcube]
            self.map.map(y, x, jac, neval_vec)
            fcn(x, fdv, neval_vec)
            
            # compute integral h-cube by h-cube
            i_start = 0
            for ihcube in range(nhcube_vec):
                hcube = hcube_base + ihcube
                sum_fdv = 0.0
                sum_fdv2 = 0.0
                for i in range(i_start, i_start + neval_hcube[ihcube]):
                    fdv[i] *= jac[i] * dv
                    fdv2[i] = fdv[i] ** 2
                    sum_fdv += fdv[i]
                    sum_fdv2 += fdv2[i]
                # following is pretty slow - 6x Kinoshita's 5-d integrand
                # sum_fdv = math.fsum(fdv[i_start:i_start + neval_hcube[ihcube]])
                # sum_fdv2 = math.fsum(fdv2[i_start:i_start + neval_hcube[ihcube]])
                mean = sum_fdv / neval_hcube[ihcube]
                sigf2 = abs(sum_fdv2 / neval_hcube[ihcube] - mean * mean)
                if redistribute:
                    self.sigf_list[hcube] = sigf2 ** (self.beta / 2.)
                var = sigf2 / (neval_hcube[ihcube] - 1.)
                ans_mean += mean
                ans_var += var
                if not adapt_to_integrand:
                    # mode = adapt_to_errors
                    tmp_hcube = hcube
                    for d in range(self.dim):
                        y0[d] = tmp_hcube % self.nstrat
                        y[0, d] = (y0[d] + 0.5) / self.nstrat
                        tmp_hcube = (tmp_hcube - y0[d]) / self.nstrat
                    fdv2[0] = var
                    self.map.add_training_data( y, fdv2, 1)
                i_start += neval_hcube[ihcube]
            if adapt_to_integrand:
                self.map.add_training_data(y, fdv2, neval_vec)

        # record final results
        self.neval_hcube_range = (min_neval_hcube, max_neval_hcube)
        # self.neval_hcube_range = (min_neval_hcube, float(counter) / nhcube, max_neval_hcube)
        return gvar.gvar(ans_mean, sqrt(ans_var))

class reporter:
    """ Analyzer class that prints out a report, iteration
    by interation, on how vegas is doing. Parameter ngrid
    specifies how many x[i]'s to print out from the maps
    for each axis.

    :param ngrid: Number of grid nodes printed out for 
        each direction. Default is 0.
    :type ngrid: int 
    """
    def __init__(self, ngrid=0):
        self.ngrid = ngrid

    def begin(self, itn, integrator):
        self.integrator = integrator
        self.itn = itn
        if itn==0:
            print(integrator.settings())

    def end(self, itn_ans, ans):
        print "    itn %2d: %s\n all itn's: %s"%(self.itn+1, itn_ans, ans)
        print(
            '    neval = %d  neval/h-cube = %s\n    chi2/dof = %.2f  Q = %.2f' 
            % (
                self.integrator.last_neval, 
                self.integrator.neval_hcube_range,
                ans.chi2 / ans.dof if ans.dof > 0 else 0,
                ans.Q if ans.dof > 0 else 1.,
                )
            )
        print(self.integrator.map.settings(ngrid=self.ngrid))
        print('')

# wrappers for scalar functions written in cython or python
# cython version comes in two flavors: with and without exceptions
# use the former where possible
cdef class VecCythonIntegrand:
    """ Vector integrand from scalar Cython integrand. """
    # cdef cython_integrand fcn
    def __init__(self): 
        self.fcntype = 'vector'

    def __call__(self, double[:, ::1] x, double[::1] f, INT_TYPE nx):
        cdef INT_TYPE i
        for i in range(nx):
            f[i] = self.fcn(x[i, :])

cdef object python_wrapper(cython_integrand fcn):
    ans = VecCythonIntegrand()
    ans.fcn = fcn
    return ans

cdef class VecCythonIntegrandExc:
    """ Vector integrand from scalar Cython integrand. """
    # cdef cython_integrand fcn
    def __init__(self): 
        self.fcntype = 'vector'

    def __call__(self, double[:, ::1] x, double[::1] f, INT_TYPE nx):
        cdef INT_TYPE i
        for i in range(nx):
            f[i] = self.fcn(x[i, :])

cdef object python_wrapper_exc(cython_integrand_exc fcn):
    ans = VecCythonIntegrandExc()
    ans.fcn = fcn
    return ans

cdef class VecPythonIntegrand:
    # cdef object fcn
    """ Vector integrand from scalar Python integrand. """
    def __init__(self, fcn):
        self.fcn = fcn
        self.fcntype = 'vector'

    def __call__(self, double[:, ::1] x, double[::1] f, INT_TYPE nx):
        cdef INT_TYPE i 
        for i in range(nx):
            f[i] = self.fcn(x[i, :])

# preferred base class for vectorized integrands
# vectorized integrands are typically faster
# der
cdef class VecIntegrand:
    """ Base class for classes providing vectorized integrands.

    A class derived from :class:`vegas.VecInterand` should
    provide a ``__call__(x, f, nx)`` member where:

        ``x[i, d]`` is a contiguous array where ``i=0...nx-1``
        labels different integrtion points and ``d=0...`` labels
        different directions in the integration space.

        ``f[i]`` is a buffer that is filled with the 
        integrand values for points ``i=0...nx-1``.

        ``nx`` is the number of integration points.

    Deriving from :class:`vegas.VecIntegrand` is the 
    easiest way to construct integrands in Cython, and
    gives the fastest results.
    """
    # cdef object fcntype
    def __cinit__(self, *args, **kargs):
        self.fcntype = 'vector'












































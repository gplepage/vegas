# c#ython: profile=True
""" Adaptive multidimensional Monte Carlo integration

Class :class:`vegas.Integrator` gives Monte Carlo estimates of 
arbitrary (square-integrable) multidimensional integrals using
the *vegas* algorithm (G. P. Lepage, J. Comput. Phys. 27(1978) 192).
It automatically remaps the integration variables along each direction
to maximize the accuracy of the Monte Carlo estimates. The remapping
is done over several iterations. The code is written in cython and
so is fast, particularly when integrands are coded in cython as
well (see *Tutorial*).

"""
cimport cython
cimport numpy
from libc.math cimport floor, log, abs, tanh, erf, exp, sqrt

import sys
import numpy 
import math
import exceptions

try:
    import gvar
    import lsqfit
    if not hasattr(lsqfit, 'WAvg'):
        # check to make sure recent version of lsqfit
        raise ImportError
except ImportError:
    # fake versions of gvar.gvar and lsqfit.wavg
    # for use if lsqfit module not available
    class _gvar_standin:
        def __init__(self):
            pass
        def gvar(self, mean, sdev):
            class GVar:
                def __init__(self, mean, sdev):
                    self.mean = float(mean)
                    self.sdev = float(sdev)
                def __str__(self):
                    return "%g +- %g" % (self.mean, self.sdev)
            return GVar(mean, sdev)

    gvar = _gvar_standin()

    class _lsqfit_standin:
        def __init__(self):
            pass
        def wavg(self, glist):
            class WAvg:
                def __init__(self, glist):
                    xmean = numpy.array([x.mean for x in glist])
                    xvar = numpy.array([x.sdev ** 2 for x in glist])
                    self.mean = numpy.sum(xmean/xvar) / numpy.sum(1./xvar)
                    self.sdev = numpy.sqrt(1. / numpy.sum(1/xvar))
                    self.chi2 = ( 
                        sum(xmean ** 2 / xvar) 
                        - self.mean ** 2 * sum(1./xvar)
                        )
                    self.dof = len(glist)
                    self.Q = -1
                def __str__(self):
                    return "%g +- %g" % (self.mean, self.sdev)
            return WAvg(glist)

    lsqfit = _lsqfit_standin()

cdef double TINY = 1e-308                            # smallest and biggest
cdef double HUGE = 1e308

# Wrapper for python functions used by Integrator
cdef object _python_integrand = None

cdef void _python_integrand_wrapper(double[:,::1] x, double[::1] f, INT_TYPE nx):
    cdef INT_TYPE i
    for i in range(nx):
        f[i] = _python_integrand(x[:, i])

cdef void _vec_python_integrand_wrapper(double[:,::1] x, double[::1] f, INT_TYPE nx):
    _python_integrand(x, f, nx)


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
    a list of ``y`` values ``y[d, i]`` spread over ``(0,1)`` for 
    each direction ``d``::

        ...
        for i in range(ny):
            for d in range(dim):
                y[d, i] = ....
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
        can be an array ``y[d,i,j,...]`` of such points (``d=0..dim-1``).
        """
        y = numpy.asarray(y, float)
        y_shape = y.shape
        y.shape = y.shape[0], -1
        x = 0 * y
        jac = numpy.empty(y.shape[1], float)
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
        y.shape = y.shape[0], -1
        x = 0 * y
        jac = numpy.empty(y.shape[1], float)
        self.map(y, x, jac)
        return jac

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef map(
        self, 
        double[:, ::1] y, 
        double[:, ::1] x, 
        double[::1] jac, 
        INT_TYPE ny=-1
        ):
        """ Map y to x, where jac is the jacobian.

        ``y[d, i]`` is an array of ``ny`` ``y``-values for direction ``d``.
        ``x[d, i]`` is filled with the corresponding ``x`` values,
        and ``jac[i]`` is filled with the corresponding jacobian 
        values. ``x`` and ``jac`` must be preallocated: for example, ::

            x = numpy.empty(y.shape, float)
            jac = numpy.empty(y.shape[1], float)

        :param y: ``y`` values to be mapped. ``y`` is a contiguous 2-d array,
            where ``y[d, i]`` contains values for points along direction ``d``.
        :type y: contiguous 2-d array of floats
        :param x: Container for ``x`` values corresponding to ``y``.
        :type x: contiguous 2-d array of floats
        :param jac: Container for jacobian values corresponding to ``y``.
        :type jac: contiguous 1-d array of floats
        :param ny: Number of ``y`` points: ``y[d, i]`` for ``d=0...dim-1``
            and ``i=0...ny-1``. ``ny`` is set to ``y.shape[0]`` if it is
            omitted (or negative).
        :type ny: int
        """
        cdef INT_TYPE ninc = self.inc.shape[1]
        cdef INT_TYPE dim = self.inc.shape[0]
        cdef INT_TYPE i, iy, d
        cdef double y_ninc, dy_ninc, tmp_jac
        if ny < 0:
            ny = y.shape[1]
        elif ny > y.shape[1]:
            raise ValueError('ny > y.shape[1]: %d > %d' % (ny, y.shape[1]))
        for i in range(ny):
            jac[i] = 1.
            for d in range(dim):
                y_ninc = y[d, i] * ninc
                iy = <int>floor(y_ninc)
                dy_ninc = y_ninc  -  iy
                if iy < ninc:
                    x[d, i] = self.grid[d, iy] + self.inc[d, iy] * dy_ninc
                    jac[i] *= self.inc[d, iy] * ninc
                else:
                    x[d, i] = self.grid[d, ninc]
                    jac[i] *= self.inc[d, ninc - 1] * ninc
        return

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef add_training_data(
        self, 
        double[:, ::1] y, 
        double[::1] f, 
        INT_TYPE ny=-1,
        ):
        """ Add training data ``f`` for ``y``-space points ``y``.

        Accumulates training data for later use by ``self.adapt()``.

        :param y: ``y`` values corresponding to the training data. 
            ``y`` is a contiguous 2-d array, where ``y[d, i]`` 
            is for points along direction ``d``.
        :type y: contiguous 2-d array of floats
        :param f: Training function values. ``f[i]`` corresponds to 
            point ``y[d, i]`` in ``y``-space.
        :type f: contiguous 2-d array of floats
        :param ny: Number of ``y`` points: ``y[d, i]`` for ``d=0...dim-1``
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
            ny = y.shape[1]
        elif ny > y.shape[1]:
            raise ValueError('ny > y.shape[1]: %d > %d' % (ny, y.shape[1]))
        for d in range(dim):
            for i in range(ny):
                iy = <int> floor(y[d, i] * ninc)
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
        return
   

cdef class Integrator(object):

    # Settings accessible via the constructor and Integrator.set
    defaults = dict(
        map=None,
        fcntype='scalar', # default integrand type
        neval=1000,       # number of evaluations per iteration
        maxinc_axis=1000,  # number of adaptive-map increments per axis
        nhcube_vec=30,    # number of h-cubes per vector
        nstrat_crit=50,   # critical number of strata
        nitn=5,           # number of iterations
        alpha=0.5,
        beta=0.75,
        mode='automatic',
        rtol=0,
        atol=0,
        analyzer=None,
        )

    def __init__(self, map, **kargs):
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
            elif k == 'nstrat_crit':
                old_val[k] = self.nstrat_crit
                self.nstrat_crit = kargs[k]
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

    def settings(self):
        cdef INT_TYPE d
        mode = self._prepare_integration()
        beta = self.beta if mode == 'adapt_to_integrand' else 0.0
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
        if beta > 0:
            ans +=   "                      redistribute points across h-cubes\n"
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
        return ans

    def _prepare_integrator(self):
        """ called by __call__ before integrating 

        Main job is to determine the number of stratifications,
        the integration mode (if automatic), and the actual number 
        of integrand evaluations to use. The decisions here 
        are mostly heuristic.
        """
        dim = self.map.dim
        neval_eff = (self.neval / 2.0) if self.beta > 0 else self.neval
        ns = int((neval_eff / 2.) ** (1. / dim))# stratifications/axis
        ni = int(self.neval / 10.)              # increments/axis
        if ni > self.maxinc_axis:
            ni = self.maxinc_axis
        if ns >= self.nstrat_crit or ns > ni:     
            new_mode = "adapt_to_errors"
            if ns > ni:
                if ns < self.maxinc_axis:
                    ni = ns
                else:
                    ns = int(ns // ni) * ni
            else:
                ni = int(ni // ns) * ns
        else:                            
            new_mode = "adapt_to_integrand"
            if ns < 1:
                ns = 1
            if ni < 1:
                ni = 1
            elif ni  > self.maxinc_axis:
                ni = self.maxinc_axis
            ni = int(ni // ns) * ns

        if self.mode == "automatic" and self.beta > 0:
            new_mode = "adapt_to_integrand"

        if new_mode == "adapt_to_errors":
            # no point in having ni > ns in this mode
            if ni > ns:
                ni = ns

        self.map.adapt(ninc=ni)    

        self.nstrat = ns
        nhcube = self.nstrat ** dim
        self.neval_hcube = int(floor(neval_eff // nhcube))
        if self.neval_hcube < 2:
            self.neval_hcube = 2
        self.dim = dim
        if nhcube < self.nhcube_vec:
            self.nhcube_vec = nhcube

        # memory allocation for work areas -- do once
        if self.beta > 0 and len(self.sigf_list) != nhcube:
            self.sigf_list = numpy.ones(nhcube, float)
        sum_neval_hcube = self.nhcube_vec * self.neval_hcube
        self._y = numpy.empty((self.dim, sum_neval_hcube), float)
        self._x = numpy.empty((self.dim, sum_neval_hcube), float)
        self._jac = numpy.empty(sum_neval_hcube, float)
        self._neval_hcube = (
            numpy.zeros(self.nhcube_vec, int) + self.neval_hcube 
            )
        self._fdv = numpy.empty(sum_neval_hcube, float)
        self._fdv2 = numpy.empty(sum_neval_hcube, float)
        return new_mode

    def _cleanup_integrator(self):
        self._y = numpy.empty((0, 0), float)
        self._x = numpy.empty((0, 0), float)
        self._jac = numpy.empty(0, float)
        self._neval_hcube = numpy.empty(0, int)
        self._fdv = numpy.empty(0, float)
        self._fdv2 = numpy.empty(0, float)

    def _prepare_integrand(self, fcn, fcntype=None):
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
        global _python_integrand 
        # determine fcntype from fcn, self or kargs 
        if 'fcntype' in kargs:
            fcntype = kargs['fcntype']
            del kargs['fcntype']
        else:
            fcntype = self.fcntype
        fcn = self._prepare_integrand(fcn, fcntype=fcntype)
        return self._integrate(fcn, kargs)

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


    cdef _integrate(self, fcn, kargs):
        if kargs:
            old_kargs = self.set(kargs)
        else:
            old_kargs = {}
        mode = self._prepare_integrator()
        self.results = [] 
        for itn in range(self.nitn):    # iterate
            if self.analyzer != None:
                self.analyzer.begin(itn, self)
            itn_ans = self._integrator(fcn, mode) 
            self.results.append(itn_ans)
            ans = lsqfit.wavg(self.results)
            if self.analyzer != None:
                self.analyzer.end(itn_ans, ans)
            self.map.adapt(alpha=self.alpha)
            if (self.rtol * abs(ans.mean) + self.atol) > ans.sdev:
                break
        self._cleanup_integrator()
        if old_kargs:
            self.set(old_kargs)               # restore settings
        return ans


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _integrator(self, fcn, mode):
        cdef INT_TYPE nhcube = self.nstrat ** self.dim 
        cdef INT_TYPE nhcube_vec = self.nhcube_vec
        cdef INT_TYPE adapt_to_integrand = (
            1 if mode == 'adapt_to_integrand' else 0
            )
        cdef INT_TYPE redistribute = (
            1 if (self.beta > 0 and adapt_to_integrand) else 0
            )
        cdef double neval_sigf = (
            self.neval / 2. / numpy.sum(self.sigf_list)
            ) if redistribute else 0
        cdef INT_TYPE[::1] y0 = numpy.empty(self.dim, int)
        cdef double dv = (1./self.nstrat) ** self.dim
        cdef double ans_mean = 0.
        cdef double ans_var = 0.
        cdef INT_TYPE sum_neval_hcube = self.nhcube_vec * self.neval_hcube
        cdef double[:, ::1] y = numpy.empty((self.dim, sum_neval_hcube), float)
        cdef double[:, ::1] x = numpy.empty((self.dim, sum_neval_hcube), float)
        cdef double[::1] jac = numpy.empty(sum_neval_hcube, float)
        cdef INT_TYPE min_neval_hcube = self.neval_hcube
        cdef INT_TYPE max_neval_hcube = self.neval_hcube
        # cdef INT_TYPE neval_hcube
        cdef INT_TYPE hcube, i, j, d, hcube_base, ihcube, i_start
        cdef INT_TYPE[:] neval_hcube = (
            numpy.zeros(self.nhcube_vec, int) + self.neval_hcube 
            )
        cdef double sum_fdv
        cdef double sum_fdv2
        cdef double[::1] fdv = numpy.empty(sum_neval_hcube, float)
        cdef double[::1] fdv2 = numpy.empty(sum_neval_hcube, float)
        cdef double sigf2
        cdef double[:, ::1] yran
        cdef INT_TYPE tmp_hcube, counter = 0 ########

        # iterate over h-cubes in batches of self.nstrat h-cubes
        # this allows for vectorization, to reduce python overhead
        self.last_neval = 0
        nhcube_vec = self.nhcube_vec
        for hcube_base in range(0, nhcube, nhcube_vec):
            if (hcube_base + nhcube_vec) > nhcube:
                nhcube_vec = nhcube - hcube_base 

            # compute neval_hcube for each h-cube
            # reinitialize work areas if necessary
            if redistribute:
                sum_neval_hcube = 0
                for ihcube in range(nhcube_vec):
                    neval_hcube[ihcube] = <int> (
                        self.sigf_list[hcube_base + ihcube] * neval_sigf
                        )
                    if neval_hcube[ihcube] < self.neval_hcube:
                        neval_hcube[ihcube] = self.neval_hcube
                    if neval_hcube[ihcube] < min_neval_hcube:
                        min_neval_hcube = neval_hcube[ihcube]
                    if neval_hcube[ihcube] > max_neval_hcube:
                        max_neval_hcube = neval_hcube[ihcube]
                    sum_neval_hcube += neval_hcube[ihcube]
                if sum_neval_hcube > y.shape[1]:
                    # memory allocation for temps if needed
                    y = numpy.empty((self.dim, sum_neval_hcube), float)  
                    x = numpy.empty((self.dim, sum_neval_hcube), float)
                    jac = numpy.empty(sum_neval_hcube, float)            
                    fdv = numpy.empty(sum_neval_hcube, float)     
                    fdv2 = numpy.empty(sum_neval_hcube, float) 
            self.last_neval += sum_neval_hcube
           
            # generate integration points and integrate
            i_start = 0
            yran = numpy.random.uniform(0., 1., (self.dim, sum_neval_hcube))
            for ihcube in range(nhcube_vec):
                hcube = hcube_base + ihcube
                tmp_hcube = hcube
                for d in range(self.dim):
                    y0[d] = tmp_hcube % self.nstrat
                    tmp_hcube = (tmp_hcube - y0[d]) / self.nstrat
                for d in range(self.dim):
                    for i in range(i_start, i_start + neval_hcube[ihcube]):
                        y[d, i] = (y0[d] + yran[d, i]) / self.nstrat
                i_start += neval_hcube[ihcube]
            self.map.map(y, x, jac, sum_neval_hcube)
            fcn(x, fdv, sum_neval_hcube)
            
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
                mean = sum_fdv / neval_hcube[ihcube]
                sigf2 = abs(sum_fdv2 / neval_hcube[ihcube] - mean * mean)
                if redistribute:
                    self.sigf_list[hcube] = sigf2 ** (self.beta / 2.)
                var = sigf2 / (neval_hcube[ihcube] - 1.)
                ans_mean += mean
                ans_var += var
                if not adapt_to_integrand:
                    for d in range(self.dim):
                        y[d, 0] = (y0[d] + 0.5) / self.nstrat
                    fdv2[0] = var
                    self.map.add_training_data( y, fdv2, 1)
                i_start += neval_hcube[ihcube]
            if adapt_to_integrand:
                self.map.add_training_data(y, fdv2, sum_neval_hcube)

        # record final results
        self.neval_hcube_range = (min_neval_hcube, max_neval_hcube)
        return gvar.gvar(ans_mean, sqrt(ans_var))

class reporter:
    """ analyzer class that prints out a report, iteration
    by interation, on how vegas is doing. Parameter n_prn
    specifies how many x[i]'s to print out from the maps
    for each axis """
    def __init__(self, n_prn=0):
        self.n_prn = n_prn

    def begin(self, itn, integrator):
        self.integrator = integrator
        self.itn = itn
        if itn==0:
            print(integrator.settings())

    def end(self, itn_ans, ans):
        print "    itn %2d: %s\n all itn's: %s"%(self.itn+1, itn_ans, ans)
        print(
            '    neval = %d  neval/h-cube = %s\n    chi2/dof = %.2f  Q = %.1f' 
            % (
                self.integrator.last_neval, 
                self.integrator.neval_hcube_range,
                ans.chi2 / ans.dof if ans.dof > 0 else 0,
                ans.Q if ans.dof > 0 else 1.,
                )
            )
        if self.n_prn > 0:
            map = self.integrator.map
            grid = numpy.array(map.grid)
            nskip = int(map.ninc // self.n_prn)

            if nskip<1:
                nskip = 1
            start = nskip // 2
            for d in range(map.dim):
                print(
                    "    grid[%2d] = %s" 
                    % (
                        d, 
                        numpy.array2string(
                            grid[d, start::nskip],precision=3,
                            prefix='    grid[xx] = ')
                          )
                    )
        print('')

cdef class VecCythonIntegrand:
    """ Vector integrand from scalar Cython integrand. """
    # cdef cython_integrand fcn
    def __init__(self): 
        self.fcntype = 'vector'

    def __call__(self, double[:, ::1] x, double[::1] f, INT_TYPE nx):
        cdef INT_TYPE i
        for i in range(nx):
            f[i] = self.fcn(x[:, i])

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
            f[i] = self.fcn(x[:, i])

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
            f[i] = self.fcn(x[:, i])

#
# Testing code:
###############

# cdef class VegasTest:
#     """ Test vegas.

#     This code creates integrands with any number of exponentials along
#     the diagonal of the integration volume, and then evalues the integral
#     using different levels of cythonizing in the code.

#     Using 3 exponentials in 8 dimensions, with order 10**6, get fastest
#     result from cython_integrate, followed by vec_integrate_cython which
#     was only 10-20% slower. The later is probably the easiest interface
#     to use for cythonized integrands. 20-30x slower are integrate_python
#     and vec_integrate_python, with the former slightly faster(!).
#     """
#     # cdef readonly double[:] sig
#     # cdef readonly double[:] x0
#     # cdef readonly double[:] ampl
#     # cdef readonly double exact

#     def __init__(
#         self, 
#         integrator,
#         x0=None, 
#         ampl=None,
#         sig=None,
#         ):
#         self.I = integrator
#         self.exact = 0.0
#         self.x0 = numpy.zeros(1, float) if x0 is None else numpy.asarray(x0)
#         self.ampl = numpy.ones(len(x0), float) if ampl is None else numpy.asarray(ampl)
#         self.sig = (0.1 + numpy.zeros(len(x0), float)) if sig is None else numpy.asarray(sig)
#         for x0, sig, ampl in zip(self.x0, self.sig, self.ampl):
#             tmp = 1.
#             for d in range(self.I.map.dim):
#                 tmp *= self.exact_gaussian(x0, sig, ampl, self.I.map.region(d))
#             self.exact += tmp

#     # @cython.boundscheck(False)
#     # @cython.wraparound(False)
#     cdef void cython_vec_fcn(self, double[:, ::1] x, double[::1] f, INT_TYPE nx):
#         cdef double ans
#         cdef double dx2, dx
#         cdef INT_TYPE i, d, j
#         cdef INT_TYPE dim = x.shape[0]
#         cdef INT_TYPE nx0 = len(self.x0)
#         for i in range(nx):
#             ans = 0.0
#             for j in range(nx0):
#                 dx2 = 0.0
#                 for d in range(dim):
#                     dx = x[d, i] - self.x0[j]
#                     dx2 += dx * dx
#                 ans += self.ampl[j] * exp(- dx2 / self.sig[j] ** 2) 
#             f[i] = ans / self.exact
#         return

#     # def python_vec_fcn(self, x, f, nx):
#     #     for i in range(nx):
#     #         ans = 0.0
#     #         for j in range(len(self.x0)):
#     #             dx2 = 0.0
#     #             for d in range(x.shape[0]):
#     #                 dx = x[d, i] - self.x0[j]
#     #                 dx2 += dx * dx
#     #             ans += self.ampl[j] * numpy.exp(- dx2 / self.sig[j] ** 2) 
#     #         f[i] = ans / self.exact
#     #     return
#     def python_vec_fcn(self, double[:, ::1] xx, double[::1] ff, nx):
#         " This is 6x faster than the version just above for one example "
#         x = numpy.asarray(xx)
#         f = numpy.asarray(ff)
#         ans = 0.0
#         for j in range(len(self.x0)):
#             dx2 = 0.0
#             for d in range(x.shape[0]):
#                 dx = x[d, :nx] - self.x0[j]
#                 dx2 += dx * dx
#             ans += self.ampl[j] * numpy.exp(- dx2 / self.sig[j] ** 2) 
#         f[:nx] = ans / self.exact
#         return

#     def python_fcn(self, x):
#         ans = 0.0
#         for j in range(len(self.x0)):
#             dx2 = 0.0
#             for d in range(x.shape[0]):
#                 dx = x[d] - self.x0[j]
#                 dx2 += dx * dx
#             ans += self.ampl[j] * numpy.exp(- dx2 / self.sig[j] ** 2) 
#         return ans / self.exact


#     # @cython.boundscheck(False)
#     # @cython.wraparound(False)
#     def __call__(self, double[:, ::1] x, double[::1] f, INT_TYPE nx):
#         cdef double ans
#         cdef double dx2, dx
#         cdef INT_TYPE i, d, j
#         cdef INT_TYPE dim = x.shape[0]
#         cdef INT_TYPE nx0 = len(self.x0)
#         for i in range(nx):
#             ans = 0.0
#             for j in range(nx0):
#                 dx2 = 0.0
#                 for d in range(dim):
#                     dx = x[d, i] - self.x0[j]
#                     dx2 += dx * dx
#                 ans += self.ampl[j] * exp(- dx2 / self.sig[j] ** 2) 
#             f[i] = ans / self.exact
#         return

#     def exact_gaussian(self, x0, sig, ampl, interval):
#         """ int dx exp(-(x-x0)**2 / sig**2) over range """
#         ans = 0.0
#         for x in [interval[0], interval[-1]]:
#             ans += erf(abs(x0 - x) / sig)
#         return ans * ampl * math.pi ** 0.5 / 2. * sig

#     def cython_integrate(VegasTest self, **kargs):
#         " Integrate using I.cython_integrate "
#         global vegastest_instance 
#         vegastest_instance = self
#         return self.I.cython_integrate(& vegastest_fcn, kargs)

#     def vec_integrate_cython(VegasTest self, **kargs):
#         " Integrate using I.vec_integrate with cython optimized python fcn "
#         return self.I.vec_integrate(self, **kargs)

#     def vec_integrate_python(self, **kargs):
#         " Integrate using I.vec_integrate with pure python vector fcn "
#         return self.I.vec_integrate(self.python_vec_fcn, **kargs)


#     def integrate_python(self, **kargs):
#         " Integrate using I(fcn) with pure python (non-vector) fcn "
#         return self.I(self.python_fcn, **kargs)


# cdef VegasTest vegastest_instance

# cdef void vegastest_fcn(double[:, ::1] x, double[::1] f, INT_TYPE nx):
#     vegastest_instance.cython_vec_fcn(x, f, nx)














































# c#ython: profile=True

cimport cython
cimport numpy
cimport cpython.array as cparray
from libc.math cimport floor, log, abs, tanh, erf, exp, sqrt

import sys
import gvar
import lsqfit
import numpy
import math
import exceptions

TINY = 1e-308                            # smallest and biggest
HUGE = 1e308

# Wrapper for python functions
cdef object _python_integrand = None

cdef void _compute_python_integrand(double[:,::1] x, double[::1] f, INT_TYPE nx):
    cdef INT_TYPE i
    for i in range(nx):
        f[i] = _python_integrand(x[:, i])

cdef void _compute_python_vec_integrand(double[:,::1] x, double[::1] f, INT_TYPE nx):
    _python_integrand(x, f, nx)
 
cdef class AdaptiveMap:
    """ Adaptive map ``y->x(y)``, where ``y`` in ``(0,1)``.

    An :class:`AdaptiveMap` defines a map ``y -> x(y)`` that is adaptive. The
    map is specified by an N-increment grid in x-space where::

        x(y) = x[i(y)] + delta_x[i(y)] * delta(y)

    with ``i(y)=floor(y*N``), ``delta(y)=y*N-i(y)``, and
    ``delta_x[i]=x[i+1]-x[i]``. The jacobian for this map is::

        dx(y)/dy = delta_x[i(y)] * N

    Each increment in the ``x``-space grid maps into an increment of 
    size ``1/N`` in the corresponding ``y`` space. So increments with 
    small ``delta_x[i]`` are stretched out in ``y`` space, while larger
    increments are shrunk.

    The ``x`` grid for an :class:`AdaptiveMap` can be specified explicitly
    when it is created: for example, ::

        map = AdaptiveMap([0., 0.1, 2.0])

    creates a map where the ``x`` interval ``(0,0.1)`` maps into the
    ``y`` interval ``(0,0.5)``, while ``(0.1,2.0)`` maps into ``(0.5,1)``
    in ``y`` space.

    More typically an initially uniform map is trained so that 
    ``F(x(y),dx(y)/dy)``, for some training function ``F``, 
    is (approximately) constant across ``y`` space. This is done 
    iteratively, beginning with a uniform map::

        m = AdaptiveMap([xl, xu], ninc=N)

    which creates an ``x`` grid with ``N`` equal-sized increments 
    between ``x=xl`` and ``x=xu``. Then training data is accumulated
    by the map by creating a list of ``y`` values spread over ``(0,1)``
    and evaluating the training function at the corresponding ``x``
    values::

        y = numpy.array([....])                # y values on (0,1)
        x = numpy.empty(len(y), float)        # container for correspondings x's 
        jac = numpy.empty(len(y), float)    # container for corresponding dx/dy's
        m.map(y=y, x=x, jac=jac)            # fill x and jac
        f = F(x, jac)                        # returns array of F values

    The training data is given to the map using::

        m.accumulate_training_data(y, f)

    This can be done all at once or in batches, with multiple calls 
    to ``m.accumulate_training_data``; the training data accumulates 
    inside the map. Finally the map is adapted to the data::

        m.adapt()

    The process of computing training data and then adapting the map
    typically has to be repeated several times before the map converges,
    at which point the ``x`` grid's nodes, ``m.x[i]``, stop changing.

    The speed with which the grid adapts is determined by 
    parameter ``alpha``. Large (positive) values imply rapid adaptation,
    while small values (much less than one) imply slow adaptation. As in
    any iterative process, it is sometimes useful to slow adaptation down
    in order to avoid instabilities.

    :param x: Initial ``x`` grid.
    :type x: 1-d sequence of floats
    :param ninc: Number of increments in ``x`` grid. New ``x`` values
        are generated if the ``ninc`` differs from ``len(x)`` for 
        parameter ``x``. These are chosen to maintain the Jacobian,
        ``dx(y)/dy``, as much as possible. Ignored if ``None`` (default).
    :type ninc: ``int`` or ``None``
    :param alpha: Determines the speed with which the grid adapts to 
        training data. Large (postive) values imply rapid evolution; 
        small values (much less than one) imply slow evolution. Typical 
        values are of order one. Choosing ``alpha<0`` causes adaptation
        to the unmodified training data (usually not a good idea).
    """
    def __cinit__(self, grid, alpha=3.0, ninc=None):
        cdef INT_TYPE i, d
        grid = numpy.array(grid, float)
        if grid.ndim != 2:
            raise ValueError('grid must be 2-d array not %d-d' % grid.ndim)
        grid.sort(axis=1)
        if grid.shape[1] < 2: 
            raise ValueError("grid.shape[1] smaller than 2: " % grid.shape[1])
        self._init_grid(grid, initinc=True)
        self.alpha = alpha
        self.sum_f = None
        self.n_f = None
        if ninc is not None and ninc != self.inc.shape[1]:
            if self.inc.shape[1] == 1:
                self.make_uniform(ninc=ninc)
            else:
                self.adapt(ninc=ninc)

    property ninc:
        def __get__(self):
            return self.inc.shape[1]
    property  dim:
        def __get__(self):
            return self.grid.shape[0]
    def region(self, INT_TYPE d=-1):
        if d < 0:
            return [self.region(d) for d in range(self.dim)]
        else:
            return (self.grid[d, 0], self.grid[d, -1])

    def make_uniform(self, ninc=None):
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
        y = numpy.asarray(y)
        y_shape = y.shape
        y.shape = y.shape[0], -1
        x = 0 * y
        jac = numpy.empty(y.shape[1], float)
        self.map(y, x, jac)
        x.shape = y_shape
        return x

    def jac(self, y):
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
        """ array y -> x,J where x and J are arrays
        where x = x(y) and J = dx/dy (jac.) """
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

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef accumulate_training_data(
        self, 
        double[:, ::1] y, 
        double[::1] f, 
        INT_TYPE ny=-1,
        ):
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
        
    def adapt(self, ninc=None, alpha=None):
        cdef double[:, ::1] new_grid
        cdef double[::1] avg_f, tmp_f
        cdef double sum_f, acc_f, f_inc
        cdef INT_TYPE old_ninc = self.grid.shape[1] - 1
        cdef INT_TYPE dim = self.grid.shape[0]
        cdef INT_TYPE i, j, new_ninc
        cdef double smth = 3.   # was 3.
        #
        # initialization
        if alpha is None:
            alpha = self.alpha
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
            self.grid = new_grid
            self._make_inc()
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
   

cdef class Integrator(object):

    defaults = dict(
        neval=1000,
        maxinc=10000,
        maxvec=10000,
        cstrat=10,
        nitn=5,
        alpha=1.5,
        mode='automatic',
        beta=0.5,
        analyzer=None,
        rtol=0,
        atol=0,
        )

    def __cinit__(self, region, **kargs):
        args = dict(Integrator.defaults)
        args.update(kargs)
        self.set(args)
        self.map = AdaptiveMap(region)
        self.sigf_list = numpy.array([], float) # dummy
        self.neval_hcube_range = None
        self.last_neval = 0

    def set(self, ka={}, **kargs):
        if kargs:
            kargs.update(ka)
        else:
            kargs = ka
        old_val = dict()
        for k in kargs:
            if k == 'neval':
                old_val[k] = self.neval
                self.neval = kargs[k]
            elif k == 'maxinc':
                old_val[k] = self.maxinc
                self.maxinc = kargs[k]
            elif k == 'maxvec':
                old_val[k] = self.maxvec
                self.maxvec = kargs[k]
            elif k == 'cstrat':
                old_val[k] = self.cstrat
                self.cstrat = kargs[k]
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

    def status(self):
        cdef INT_TYPE d
        self._prep_integration()
        nhcube = self.nstrat ** self.dim
        neval = nhcube * self.neval_hcube
        ans = ""
        ans = "Integrator Status:\n"
        if self.beta > 0:
            ans = ans + (
                "    %d (approx) integrand evaluations in each of %d iterations\n"
                % (self.neval, self.nitn)
                )
        else:
            ans = ans + (
                "    %d integrand evaluations in each of %d iterations\n"
                % (neval, self.nitn)
                )
        mode = self._mode
        if self.mode=="automatic":
            mode = self._mode + "  (automatic)"
        ans = ans + ("    integrator mode = %s\n" % mode)
        if self.beta > 0:
            ans +=   "                      redistribute points across h-cubes\n"
        ans = ans + (
            "    number of:  strata/axis = %d  increments/axis = %d\n"
            % (self.nstrat, self.map.ninc)
            )
        if self.beta > 0:
            ans = ans + (
                "                h-cubes = %d  evaluations/h-cube = %d (min)\n"
                % (nhcube, self.neval_hcube)
                )
        else:
            ans = ans + (
                    "                h-cubes = %d  evaluations/h-cube = %d\n"
                    % (nhcube, self.neval_hcube)
                    )
        n_vec = min(max(nhcube, self.neval_hcube), self.maxvec)
        ans = ans + (
            "    damping parameters: alpha = %g  beta= %g\n" 
            % (self.alpha, self.beta)
            )
        ans = ans + ("    accuracy: relative = %g" % self.rtol)
        ans = ans + ("  absolute accuracy = %g\n\n" % self.atol)
        for d in range(self.dim):
            ans = ans +(
                "    axis %d covers %s\n" % (d, str(self.map.region(d)))
                )
        return ans

    def _prep_integration(self):
        """ called by __call__ before integrating """
        dim = self.map.dim
        neval_eff = self.neval / 1.5 if self.beta > 0 else self.neval
        ns = int((neval_eff / 2.) ** (1. / dim))     # stratifications/axis
        if ns > self.maxvec:
            ns = self.maxvec
        ni = int(self.neval / 10.)                    # increments/axis
        if ns > self.cstrat or ns > ni:     
            self._mode = "adapt_to_errors"
            ni = ns
            if ni * dim > self.maxinc:
                ni = int(self.maxinc / dim)
                ns = int(ns / ni) * ni
        else:                            
            self._mode = "adapt_to_integrand"
            if ns < 1:
                ns = 1
            if ni < 1:
                ni = 1
            elif ni * dim > self.maxinc:
                ni = int(self.maxinc / dim)
            ni = int(ni // ns) * ns

        if self.mode != "automatic":
            self._mode = self.mode
        
        self.map.adapt(ninc=ni)    
        self.map.alpha = self.alpha

        self.nstrat = ns
        nhcube = self.nstrat ** dim
        self.neval_hcube = int(floor(self.neval // nhcube))
        if self.neval_hcube < 2:
            self.neval_hcube = 2
        self.dim = dim
        if self.beta > 0 and len(self.sigf_list) != nhcube:
            self.sigf_list = numpy.ones(nhcube, float)

    def integrate(self, fcn, **kargs):
        global _python_integrand    
        self.vec_integrand = _compute_python_integrand
        _python_integrand = fcn
        return self._integrate(kargs)

    def __call__(self, fcn, **kargs):
        return self.integrate(fcn, **kargs)

    def vec_integrate(self, fcn, **kargs):
        global _python_integrand    
        self.vec_integrand = _compute_python_vec_integrand
        _python_integrand = fcn
        return self._integrate(kargs)

    cdef cython_integrate(self, cython_vec_integrand fcn, kargs):
        self.vec_integrand = fcn
        return self._integrate(kargs)

    def test_cython_integrate(self, **kargs):
        return self.cython_integrate(& vec_example, kargs)

    cdef _integrate(self, kargs):
        if kargs:
            old_kargs = self.set(kargs)
        else:
            old_kargs = {}
        self._prep_integration()
        anslist = [] 
        for itn in range(self.nitn):    # iterate
            if self.analyzer != None:
                self.analyzer.begin(itn, self)
            itn_ans = self._integrator() 
            anslist.append(itn_ans)
            ans = lsqfit.wavg(anslist)
            if self.analyzer != None:
                self.analyzer.end(itn_ans, ans)
            self.map.adapt()
            if (self.rtol * abs(ans.mean) + self.atol) > ans.sdev:
                break
        if old_kargs:
            self.set(old_kargs)               # restore settings
        return ans


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _integrator(self):
        cdef INT_TYPE nhcube = self.nstrat ** self.dim 
        cdef INT_TYPE adapt_to_integrand = (
            1 if self._mode == 'adapt_to_integrand' else 0
            )
        cdef INT_TYPE redistribute = (
            1 if self.beta > 0 else 0
            )
        cdef double max_sigf = numpy.max(self.sigf_list) if redistribute else 0
        cdef double sum_sigf = numpy.sum(self.sigf_list) if redistribute else 0
        cdef double neval_factor = (
            max_sigf * nhcube * self.neval_hcube / sum_sigf
            ) if redistribute else 0
        cdef INT_TYPE[::1] y0 = numpy.empty(self.dim, int)
        cdef double dv = (1./self.nstrat) ** self.dim
        cdef double ans_mean = 0.
        cdef double ans_var = 0.
        cdef double[:, ::1] y = numpy.empty((0,0), float)
        cdef double[:, ::1] x = numpy.empty((0,0), float)
        cdef double[::1] jac = numpy.empty((0,), float)
        cdef INT_TYPE min_neval_hcube = self.neval_hcube
        cdef INT_TYPE max_neval_hcube = self.neval_hcube
        cdef INT_TYPE neval_hcube
        cdef INT_TYPE hcube, i, j, d
        cdef double sum_fdv
        cdef double sum_fdv2
        cdef double[::1] fdv = numpy.empty((0,), float)
        cdef double[::1] fdv2 = numpy.empty((0,), float)
        cdef double sigf2, sigf
        cdef double[:, ::1] yran
        cdef INT_TYPE tmp_hcube

        self.last_neval = 0
        for hcube in range(nhcube):
            tmp_hcube = hcube
            for d in range(self.dim):
                y0[d] = tmp_hcube % self.nstrat
                tmp_hcube = (tmp_hcube - y0[d]) / self.nstrat
            if redistribute:
                neval_hcube = <int> (
                    tanh(self.sigf_list[hcube] / max_sigf) * neval_factor
                    )
                if neval_hcube < self.neval_hcube:
                    neval_hcube = self.neval_hcube
                if neval_hcube < min_neval_hcube:
                    min_neval_hcube = neval_hcube
                if neval_hcube > max_neval_hcube:
                    max_neval_hcube = neval_hcube
            else:
                neval_hcube = self.neval_hcube
            self.last_neval += neval_hcube
            if neval_hcube > y.shape[1]:
                # memory allocation for temps if needed
                y = numpy.empty((self.dim, neval_hcube), float)  
                x = numpy.empty((self.dim, neval_hcube), float)
                jac = numpy.empty(neval_hcube, float)            
                fdv = numpy.empty(neval_hcube, float)     
                fdv2 = numpy.empty(neval_hcube, float) 
            yran = numpy.random.uniform(0., 1., (self.dim, neval_hcube))
            for d in range(self.dim):
                for i in range(neval_hcube):
                    y[d, i] = (y0[d] + yran[d, i]) / self.nstrat
            self.map.map(y, x, jac, neval_hcube)
            sum_fdv = 0.0
            sum_fdv2 = 0.0
            self.vec_integrand(x, fdv, neval_hcube)
            for i in range(neval_hcube):
                # fdv[i] = (<double> fcn(x[:, i])) * jac[i] * dv     ##
                # fdv[i] = example(x[:, i]) * jac[i] * dv     ##
                fdv[i] *= jac[i] * dv
                fdv2[i] = fdv[i] ** 2
                sum_fdv += fdv[i]
                sum_fdv2 += fdv2[i]
            mean = sum_fdv / neval_hcube
            sigf2 = abs(sum_fdv2 / neval_hcube - mean * mean)
            if redistribute:
                if True:
                    self.sigf_list[hcube] = (
                        # self.sigf_list[hcube] ** (1 - self.beta) *
                        sqrt(sigf2) ** self.beta
                        ) 
                else:
                    sigf = sqrt(sigf2)
                    self.sigf_list[hcube] = (-(1 - sigf) / log(sigf)) ** self.beta
            var = sigf2 / (neval_hcube - 1.)
            ans_mean += mean
            ans_var += var  
            if adapt_to_integrand:
                self.map.accumulate_training_data(y, fdv2, neval_hcube)
            else:
                for d in range(self.dim):
                    y[d, 0] = (y0[d] + 0.5) / self.nstrat
                fdv2[0] = var
                self.map.accumulate_training_data( y, fdv2, 1)
        self.neval_hcube_range = (min_neval_hcube, max_neval_hcube)
        return gvar.gvar(ans_mean, sqrt(ans_var))


class reporter:
    """ analyzer class that prints out a report, iteration
    by interation, on how vegas is doing. Parameter n_prn
    specifies how many x[i]'s to print out from the maps
    for each axis """
    def __init__(self, n_prn=5):
        self.n_prn = n_prn

    def begin(self, itn, integrator):
        self.integrator = integrator
        self.itn = itn
        if itn==0:
            print(integrator.status())

    def end(self, itn_ans, ans):
        print "    itn %2d: %s\n all itn's: %s"%(self.itn+1, itn_ans, ans)
        print(
            '    chi2/dof = %.2f  neval = %d  neval/h-cube = %s' 
            % (
                ans.chi2/ans.dof if ans.dof > 0 else 0, 
                self.integrator.last_neval, 
                self.integrator.neval_hcube_range,
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




#
# Testing code:
###############

cdef class test_integrand:
    # cdef readonly double[:] sig
    # cdef readonly double[:] x0
    # cdef readonly double[:] ampl
    # cdef readonly double exact

    def __init__(
        self, 
        region,
        x0=None, 
        ampl=None,
        sig=None,
        ):
        self.exact = 0.0
        self.x0 = numpy.zeros(len(region), float) if x0 is None else numpy.asarray(x0)
        self.ampl = numpy.ones(len(region), float) if ampl is None else numpy.asarray(ampl)
        self.sig = (0.1 + numpy.zeros(len(region), float)) if sig is None else numpy.asarray(sig)
        for x0, sig, ampl in zip(self.x0, self.sig, self.ampl):
            tmp = 1.
            for r in region:
                tmp *= self.exact_gaussian(x0, sig, ampl, r)
            self.exact += tmp

    def exact_gaussian(self, x0, sig, ampl, interval):
        """ int dx exp(-(x-x0)**2 / sig**2) over range """
        ans = 0.0
        for x in [interval[0], interval[-1]]:
            ans += erf(abs(x0 - x) / sig)
        return ans * ampl * math.pi ** 0.5 / 2. * sig

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    cdef void fcn(self, double[:, ::1] x, double[::1] f, INT_TYPE nx):
        cdef double ans
        cdef double dx2, dx
        cdef INT_TYPE i, d, j
        cdef INT_TYPE dim = x.shape[0]
        cdef INT_TYPE nx0 = len(self.x0)
        for i in range(nx):
            ans = 0.0
            for j in range(nx0):
                dx2 = 0.0
                for d in range(dim):
                    dx = x[d, i] - self.x0[j]
                    dx2 += dx * dx
                ans += self.ampl[j] * exp(- dx2 / self.sig[j] ** 2) 
            f[i] = ans / self.exact
        return

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    def __call__(self, double[:, ::1] x, double[::1] f, INT_TYPE nx):
        cdef double ans
        cdef double dx2, dx
        cdef INT_TYPE i, d, j
        cdef INT_TYPE dim = x.shape[0]
        cdef INT_TYPE nx0 = len(self.x0)
        for i in range(nx):
            ans = 0.0
            for j in range(nx0):
                dx2 = 0.0
                for d in range(dim):
                    dx = x[d, i] - self.x0[j]
                    dx2 += dx * dx
                ans += self.ampl[j] * exp(- dx2 / self.sig[j] ** 2) 
            f[i] = ans / self.exact
        return

# @cython.boundscheck(False)
# @cython.wraparound(False)
dim = 8
nexp = 3
region = dim * [[0., 1.]]
sig = nexp * [0.1]
x0 = numpy.linspace(0., 1., nexp + 2)[1:-1] 
cdef test_integrand tester_fcn = test_integrand(region=region, x0=x0, sig=sig)

cdef void vec_example(double[:, ::1] x, double[::1] f, INT_TYPE nx):
    tester_fcn.fcn(x, f, nx)
    # cdef INT_TYPE i
    # for i in range(nx):
    #     f[i] = exp(-100. * (x[0, i] * x[0, i] + x[1, i] * x[1, i])) * 100. / 3.14159654

cdef double example(double[:] x):
        return exp(-100. * (x[0] * x[0] + x[1] * x[1])) * 100. / 3.14159654



                















































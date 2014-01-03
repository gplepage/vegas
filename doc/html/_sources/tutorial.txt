Tutorial
=======================================

.. moduleauthor:: G. Peter Lepage <g.p.lepage@cornell.edu>

.. |Integrator| replace:: :class:`vegas.Integrator`
.. |AdaptiveMap| replace:: :class:`vegas.AdaptiveMap`
.. |vegas| replace:: :mod:`vegas`
.. |WAvg| replace:: :class:`vegas.RunningWAvg`
.. |chi2| replace:: :math:`\chi^2`
.. |x| replace:: x 
.. |y| replace:: y 

Introduction
-------------

Class :class:`vegas.Integrator` gives Monte Carlo estimates of arbitrary
multidimensional integrals using the *vegas* algorithm 
(G. P. Lepage, J. Comput. Phys. 27 (1978) 192).  
The algorithm has two components.
First an automatic transformation is applied to to the integration variables
in an attempt to flatten the integrand. Then a Monte Carlo estimate of the
integral is made using the  transformed variables. Flattening the integrand
makes the integral easier and improves the estimate.  The transformation
applied to the integration variables is optimized
over several iterations of the algorithm: information about the integrand that
is collected during one iteration is used to  improve the transformation used
in the next iteration.

Monte Carlo integration makes few assumptions about the 
integrand --- it needn't be analytic nor even continuous. This
makes Monte Carlo integation unusually robust. It also makes it well suited
for adaptive integration. Adaptive strategies are essential for
multidimensional integration, especially in high dimensions, because
multidimensional space is large, with  lots of corners, making it 
easy to lose important features in the integrand. 

Monte Carlo integration also provides efficient and reliable methods for
estimating the 
accuracy of its results. In particular, each Monte Carlo
estimate of an integral is a random number from a distribution
whose mean is the correct value of the integral. This distribution is
Gaussian or normal provided 
the number of integrand samples is sufficiently large. 
In practive one generates multiple
estimates of the integral
in order to verify that the distribution is indeed Gaussian. 
Error analysis is straightforward if the
integral estimates are Gaussian.

The |vegas| algorithm has been in use for decades and implementations are
available in may programming languages, including Fortran (the original
version), C and C++. The algorithm used here is significantly improved over
the original implementation, and that used in most other implementations.
It uses two adaptive strategies: importance sampling, as in the original
implementation, and adaptive stratified sampling, which is new.

This module is written in Cython, so it is almost as fast as optimized Fortran or
C, particularly when the integrand is also coded in Cython (or some other
compiled language), as discussed below.

*About Printing:* The examples in this tutorial use the print function as it is
used in Python 3. Drop the outermost parenthesis in each print statement if
using Python 2, or add ::

    from __future__ import print_function

at the start of your file.


Basic Integrals
----------------
Here we illustrate the use of |vegas| by estimating the integral

.. math::

    C\int_{-1}^1 dx_0 \int_0^1 dx_1 \int_0^1 dx_2 \int_0^1 dx_3
    \,\,\mathrm{e}^{- 100 \sum_{d}(x_d-0.5)^2}  ,

where constant :math:`C` is chosen so that the exact integral is 1. 
The following code shows how this can be done::

    import vegas
    import math

    def f(x): 
        dx2 = 0 
        for d in range(4): 
            dx2 += (x[d] - 0.5) ** 2
        return math.exp(-dx2 * 100.) * 1013.2118364296088

    integ = vegas.Integrator([[-1., 1.], [0., 1.], [0., 1.], [0., 1.]])

    result = integ(f, nitn=10, neval=1000)
    print(result.summary())
    print('result = %s    Q = %.2f' % (result, result.Q))

First we define the integrand ``f(x)`` where ``x[d]`` specifies a  point in the
4-dimensional space. We then create an  integrator, ``integ``, which is an
integration operator  that can be applied to any 4-dimensional function. It is
where we specify the integration volume. 
Finally we apply ``integ`` to our integrand ``f(x)``,
telling the integrator to estimate  the integral using ``nitn=10`` iterations
of the |vegas| algorithm, each of which uses no more than ``neval=1000``
evaluations of the integrand. Each iteration produces an independent
estimate of the integral. The final estimate is the weighted average of
the results from all 10 iterations, and is returned by ``integ(f ...)``.
The call ``result.summary()`` returns 
a summary of results from each iteration.

This code produces the following output:

.. literalinclude:: eg1a.out

There are several things to note here:

    **Adaptation:** Integration estimates are shown for
    each of the 10 iterations,
    giving both the estimate from just that iteration, and the weighted
    average of results from all iterations up to that point. The
    estimates from the first two iterations are not accurate at
    all, with errors equal to 30--190% of the final result. 
    |vegas| initially has no information about the integrand
    and so does a relatively poor job of estimating the integral.
    It uses information from the samples in one iteration, however,
    to remap the integration variables for subsequent iterations,
    concentrating samples where the function is largest and reducing 
    errors. 
    As a result, the per-iteration error
    is reduced to 3.4% by the fifth iteration, and below 2% by
    the end --- an improvement by almost two orders of 
    magnitude from the start. Eventually the per-iteration error
    stops decreasing because |vegas| has found the optimal remapping, at which
    point
    it has fully adapted to the integrand.

    **Weighted Average:** The final result, 1.0015 ± 0.0091, 
    is obtained from a weighted
    average of the separate results from each iteration. 
    The individual estimates are statistical: each
    is a random number drawn from a distribution whose mean
    equals the correct value of the integral, and the errors 
    quoted are estimates of the standard deviations of those
    distributions. The distributions are Gaussian provided 
    the number of integrand evaluations per iteration (``neval``)
    is sufficiently large, in which case the standard deviation
    is a reliable estimate of the error.
    The weighted average :math:`\overline I`  minimizes

    .. math::

      \chi^2 \,\equiv\, \sum_i \frac{(I_i - \overline I)^2}{\sigma_{i}^2}

    where :math:`I_i \pm \sigma_{i}` are the estimates from 
    individual iterations. If the :math:`I_i` are Gaussian, 
    :math:`\chi^2` should be of order the number of degrees of 
    freedom (plus or minus the square root of that number);
    here the number of degrees of freedom is the number of 
    iterations minus 1. 

    The distributions are likely non-Gaussian, and error estimates
    unreliable, if |chi2| is
    much larger than the number of iterations. This criterion is quantified
    by the *Q* or *p-value* of the :math:`\chi^2`,
    which is the probability that a
    larger :math:`\chi^2` could result from random (Gaussian)
    fluctuations. A very small *Q* (less than 0.05-0.1) indicates
    that the :math:`\chi^2` is too large to be accounted for by
    statistical fluctuations --- that is, the estimates of the integral
    from different iterations do not agree with each other to 
    within errors. This means that ``neval`` is not sufficiently
    large to guarantee Gaussian behavior, and must be increased
    if the error estimates are to be trusted.


    ``integ(f...)`` returns a weighted-average object,
    of type :class:`vegas.RunningWAvg`, that has the following 
    attributes:

      ``result.mean`` --- weighted average of all estimates of the integral;
      
      ``result.sdev`` --- standard deviation of the weighted average;
      
      ``result.chi2`` --- :math:`\chi^2` of the weighted average;

      ``result.dof`` --- number of degrees of freedom;

      ``result.Q`` --- *Q* or *p-value* of the weighted average's |chi2|;

      ``result.itn_results`` --- list of the integral estimates 
      from each iteration.

    In this example the final *Q* is 0.42, indicating that the
    :math:`\chi^2` for this average is not particularly unlikely and
    thus the error estimate is most likely reliable.

    **Precision:** The precision of |vegas| estimates is
    determined by ``nitn``, the number of iterations 
    of the |vegas| algorithm,
    and by ``neval``, the maximum number of integrand evaluation
    made per iteration.
    The computing cost is typically proportional to the
    product of ``nitn`` and ``neval``. 
    The number of integrand
    evaluations per iteration
    varies from iteration to iteration,
    here between 486 and 959. Typically |vegas| needs more
    integration points in early iterations, before it has fully
    adapted to the integrand.

    We can increase precision by increasing either ``nitn`` or ``neval``,
    but it is 
    generally far better to increase ``neval``. For example,
    adding the following lines to the code above ::

      result = integ(f, nitn=100, neval=1000)
      print('larger nitn  => %s    Q = %.2f' % (result, result.Q))
      
      result = integ(f, nitn=10, neval=1e4)
      print('larger neval => %s    Q = %.2f' % (result, result.Q))

    generates the following results:

    .. literalinclude:: eg1b.out

    The total number of integrand evaluations, ``nitn * neval``, is
    about the same in both cases, but increasing ``neval`` is more
    than twice as accurate as increasing ``nitn``. Typically one
    wants to use no more than 10 or 20 iterations beyond the
    point where vegas has fully adapted. You want some number of
    iterations so that you can verify Gaussian behavior by 
    checking the |chi2| and *Q*, but not too many. 

    It is also generally useful to compare two or more 
    results from values of ``neval`` that differ by a
    significant factor (4--10, say). These should agree within
    errors. If they do not, it could be due to non-Gaussian
    artifacts caused by a small ``neval``. |vegas| 
    estimates have two sources of error. One is the statistical
    error, which is what is quoted by |vegas|. The other is 
    a systematic error due to residual non-Gaussian 
    effects. The systematic error vanishes like
    ``1/neval`` and so becomes negligible compared with 
    the statistical error as ``neval`` increases. 
    The systematic error can bias the Monte Carlo estimate, however, 
    if ``neval`` is insufficiently large. This usually 
    results in a large |chi2| (and small *Q*), but a 
    more reliable check is to compare
    results that use signficantly different values of ``neval``.
    The systematic errors due to non-Gaussian behavior are
    likely negligible if the different estimates agree to
    within the statistical errors. 

    The possibility of systematic biases
    is another reason for increasing ``neval`` 
    rather than ``nitn`` to obtain more precision. 
    Making ``neval`` larger and larger is guaranteed
    to improve the Monte Carlo estimate, as the statistical
    error decreases (at least as fast as ``sqrt(1/neval)``
    and often faster) and the 
    systematic error decreases even more quickly (like
    ``1/neval``). 
    Making ``nitn`` larger and larger, on the other hand,
    is guaranteed eventually to give the wrong
    answer. This is because at some point the statistical error 
    (which falls as ``sqrt(1/nitn)``) will no longer
    mask the systematic error (which is unaffected by ``nitn``). 
    The systematic error for the integral
    above (with ``neval=1000``) is about -0.00073(7), which 
    is negligible compared to the statistical error unless
    ``nitn`` is of order 1500 or larger --- so systematic errors
    aren't a problem with ``nitn=10``.

    **Early Iterations:** Integral estimates from early iterations, 
    before |vegas| has adapted, can be quite 
    crude. With very peaky integrands, these are often far from 
    the correct answer with highly unreliable error estimates. For 
    example, the integral above becomes more 
    difficult if we double the length of each side of the 
    integration volume by redefining ``integ`` as::

      integ = vegas.Integrator(
        [[-2., 2.], [0, 2.], [0, 2.], [0., 2.]],
        )

    The code above then gives:

    .. literalinclude:: eg1c.out

    |vegas| misses the peak completely in the first two iterations,
    giving estimates that are completely 
    wrong (by 76 and 89 standard deviations!).
    Some of its samples hit the peak's shoulders, so |vegas| is 
    eventually able to find the peak (by iterations 5--6), but 
    the integrand estimates are wildly non-Gaussian before that
    point. This results in a non-sensical final result, as 
    indicated by the ``Q = 0.00``. 

    It is common practice in using |vegas| to discard 
    estimates from the first several iterations, before the 
    algorithm has adapted, in order to avoid ruining the 
    final result in this way. This is done by replacing the 
    single call to ``integ(f...)`` in the original code 
    with two calls::

      # step 1 -- adapt to f; discard results
      integ(f, nitn=7, neval=1000)

      # step 2 -- integ has adapted to f; keep results
      result = integ(f, nitn=10, neval=1000)
      print(result.summary())
      print('result = %s    Q = %.2f' % (result, result.Q))

    The integrator is trained in the first 
    step, as it adapts to the integrand, and so is more or less
    fully adapted from the start in the second step, which yields:

    .. literalinclude:: eg1d.out

    The final result is now reliable.

    **Other Integrands:** Once ``integ`` has been trained on ``f(x)``, 
    it can be usefully applied
    to other functions with similar structure. For example, adding
    the following at the end of the original code, ::

      def g(x):
          return x[0] * f(x)

      result = integ(g, nitn=10, neval=1000)
      print(result.summary())
      print('result = %s    Q = %.2f' % (result, result.Q))

    gives the following new output:

    .. literalinclude:: eg1e.out

    Again the grid is almost optimal for ``g(x)`` from the start,
    because ``g(x)`` peaks in the same region as ``f(x)``.
    The exact value for this integral is very close to 0.5.

    Note that |Integrator|\s can be saved in files and reloaded later using
    Python's :mod:`pickle` module: for example, 
    ``pickle.dump(integ, openfile)`` saves integrator ``integ``
    in file ``openfile``, and 
    ``integ = pickle.load(openfile)`` reloads it. The is useful for costly
    integrations that might need to be reanalyzed later since the integrator
    remembers the variable transformations made to minimize errors, and
    so need not be readapted to the integrand when used later.

    **Non-Rectangular Volumes:** |vegas| can integrate over volumes of 
    non-rectangular shape. For example, we can replace integrand ``f(x)`` 
    above 
    by the same Gaussian, but restricted to a 4-sphere of radius 0.2,
    centered on the Gaussian::

        import vegas 
        import math 

        def f_sph(x):
            dx2 = 0 
            for d in range(4): 
                dx2 += (x[d] - 0.5) ** 2
            if dx2 < 0.2 ** 2:
                return math.exp(-dx2 * 100.) * 1115.3539360527281318
            else:
                return 0.0

        integ = vegas.Integrator([[-1., 1.], [0., 1.], [0., 1.], [0., 1.]])

        integ(f_sph, nitn=10, neval=1000)           # adapt the grid
        result = integ(f_sph, nitn=10, neval=1000)  # estimate the integral
        print(result.summary())
        print('result = %s    Q = %.2f' % (result, result.Q))
 

    The normalization is adjusted to again make the 
    exact integral equal 1. Integrating as before gives:

    .. literalinclude:: eg1f.out

    It is a good idea to make the actual integration volume as large a 
    fraction as possible of the total volume used by |vegas| ---
    by choosing integration variables properly --- so 
    |vegas| doesn't spend lots of effort on regions where the integrand
    is exactly 0. Also, it can be challenging for |vegas|
    to find the region of 
    non-zero integrand in high dimensions: integrating ``f_sph(x)``
    in 20 dimensions instead of 4, for example, 
    would require ``neval=1e16`` 
    integrand evaluations per iteration to have any chance of 
    finding the region of non-zero integrand, because the volume of 
    the 20-dimensional sphere is a tiny fraction of the total 
    integration volume. 

    Note, finally, that integration to infinity is also possible:
    map the relevant variable into a different variable
    of finite range. For example,  an integral over :math:`x\equiv\tan(\theta)`
    from 0 to infinity is easily reexpressed as 
    an integral over :math:`\theta` from 0 to :math:`\pi/2`.

    **Damping:** This result in the previous section 
    can be improved somewhat by slowing down
    |vegas|’s adaptation::

        ...
        integ(f_sph, nitn=10, neval=1000, alpha=0.1)
        result = integ(f_sph, nitn=10, neval=1000, alpha=0.1)  
        ...

    Parameter ``alpha`` controls the speed with which |vegas|
    adapts, with smaller ``alpha``\s giving slower adaptation. 
    Here we reduce ``alpha`` to 0.1, from its default value of 0.5, and get
    the following output:

    .. literalinclude:: eg1g.out

    Notice how the errors fluctuate less from iteration to iteration
    with the smaller ``alpha`` in this case. 
    Persistent, large fluctuations in the size 
    of the per-iteration errors is often a signal that ``alpha`` should
    be reduced. With larger ``alpha``\s, |vegas| can over-react
    to random fluctuations it encounters as it samples the integrand. 

    In general, we want ``alpha`` to be large enough so that |vegas| adapts
    quickly to the integrand, but not so large that it has difficulty 
    holding on to the optimal tuning once it has found it. The best value
    depends upon the integrand. Adaptation can be turned off completely
    by setting parameter ``adapt=False``: e.g., ::

        ...
        integ(f_sph, nitn=10, neval=1000, alpha=0.1)
        result = integ(f_sph, nitn=10, neval=1000, adapt=False)  
        ...


Faster Integrands
-------------------------
The computational cost of a realistic multidimensional integral 
comes mostly from
the cost of evaluating the integrand at the Monte Carlo sample 
points. Integrands written in pure Python are probably fast 
enough for problems where ``neval=1e3`` or ``neval=1e4`` gives
enough precision. Some problems, however, require
hundreds of thousands or millions of function evaluations, or more.

The cost of evaluating the integrand can be reduced significantly
by vectorizing it, if that is possible. For example,
replacing ::

    import vegas
    import math

    dim = 4
    norm = 1013.2118364296088

    def f_scalar(x):              
        dx2 = 0.0
        for d in range(dim):
            dx2 += (x[d] - 0.5) ** 2
        return math.exp(-100. * dx2) * norm

    integ = vegas.Integrator(dim * [[0, 1]])

    integ(f_scalar, nitn=10, neval=200000)
    result = integ(f_scalar, nitn=10, neval=200000)
    print('result = %s   Q = %.2f' % (result, result.Q))



by ::

    import vegas
    import numpy as np

    dim = 4

    class f_vector(vegas.VecIntegrand):
        def __init__(self, dim):
            self.dim = dim
            self.norm = 1013.2118364296088


        def __call__(self, x, f, nx):
            # convert integration points x[i, d] to numpy array
            x = np.asarray(x)[:nx, :]
            
            # convert array for answer into a numpy array
            f = np.asarray(f)[:nx]
            
            # evaluate integrand for all values of i simultaneously
            dx2 = 0.0
            for d in range(self.dim):
                dx2 += (x[:, d] - 0.5) ** 2

            # copy answer into f (ie, don't use f = np.exp(...))
            f[:] = np.exp(-100. * dx2) * self.norm

    integ = vegas.Integrator(dim * [[0, 1]], nhcube_vec=1000)

    f = f_vector(dim=dim)
    integ(f, nitn=10, neval=200000)
    result = integ(f, nitn=10, neval=200000)
    print('result = %s   Q = %.2f' % (result, result.Q))

reduces the cost of the integral by about an order of magnitude. 
An instance of class ``f_vector`` behaves like a function of
three variables:

    ``x[i, d]`` --- integration points for each ``i=0...nx-1``
    (``d=0...`` labels the direction);

    ``f[i]`` --- buffer to hold the integrand values 
    for each integration point;

    ``nx`` --- number of integration points.

We derive class ``f_vector`` from :class:`vegas.VecIntegrand` to 
signal to |vegas| that it should present integration points in 
batches to the integrand function. Parameter ``nhcube_vec`` tells
|vegas| how many hypercubes to put in a batch (or vector); the bigger 
this parameter is, the larger the batches. 

Unfortunately many problems are difficult to 
vectorize. The fastest option in such cases (and actually
every case) is to write the integrand in Cython, which
is a compiled hybrid of Python and C. The Cython version
of this code, which we put in a separate file we
call ``cython_integrand.pyx``, is simpler than the vector version::

    cimport vegas                   # for VecIntegrand
    from libc.math cimport exp      # use exp() from C library

    import vegas

    cdef class f_cython(vegas.VecIntegrand):
        cdef double norm
        cdef int dim

        def __init__(self, dim):
            self.dim = dim
            self.norm = 1013.2118364296088 ** (dim / 4.)

        def __call__(self, double[:, ::1] x, double[::1] f, int nx):
            cdef int i, d
            cdef double dx2
            for i in range(nx):
                dx2 = 0.0
                for d in range(self.dim):
                    dx2 += (x[i, d] - 0.5) ** 2
                f[i] = exp(-100. * dx2) * self.norm
            return

The main code is then ::

    import pyximport; pyximport.install()

    import vegas
    from cython_integrand import f_cython

    dim = 4

    integ = vegas.Integrator(dim * [[0, 1]], nhcube_vec=1000)

    f = f_cython(dim=dim)
    integ(f, nitn=10, neval=200000)
    result = integ(f, nitn=10, neval=200000)
    print('result = %s   Q = %.2f' % (result, result.Q))

where the first line (``import pyximport; ...``) causes the Cython
module ``cython_integrand.pyx`` to be compiled the first time
it is called. The compiled code is stored and used in subsequent 
calls, so compilation occurs only once.

Cython code can also link easily to compiled C or Fortran code, 
so integrands written in these languages can be used as well (and
would be faster than pure Python).

Multiple Integrands Simultaneously
-----------------------------------
|vegas| can be used to integrate multiple integrands simultaneously, using
the same integration points for each of the integrands. This is useful 
in situations where the integrands have similar structure, with peaks in
the same locations. There can be a very signficant advantage in sampling
the different integrands at precisely the same points in |x| space, because
the Monte Carlo estimates for the different integrals are then correlated. 
If the integrands are very similar to each other the correlations can be 
very strong, leading to greatly reduced errors in ratios or differences
of the resulting integrals. |vegas| captures these correlations by 
examing fluctuations in estimates of the different integrals over, 
typically, 10--20 iterations of the |vegas| algorithm with adaptation 
turned off. (It is important to turn off adaptation so that estimates from
different iterations come from the same probability distribution.)

Consider a simple example. We want to compute 
the normalization and first two moments of a 
sharply peaked probability distribution:

.. math::
    I_0 &\equiv \int_0^1 d^4x\;
        \mathrm{e}^{- 200 \sum_{d}(x_d-0.5)^2}\\
    I_1 &\equiv \int_0^1 d^4x\; x_0 \; 
        \mathrm{e}^{- 200 \sum_{d}(x_d-0.5)^2} \\
    I_2 &\equiv \int_0^1 d^4x\; x_0^2 \; 
        \mathrm{e}^{- 200 \sum_{d}(x_d-0.5)^2}

From these integrals we can determine the mean and width of the distribution
projected onto one of the axes: 

.. math::
    \langle x \rangle &\equiv I_1 / I_0 \\[1ex]
    \sigma_x^2 &\equiv \langle x^2 \rangle - \langle x \rangle^2 \\
               &= I_2 / I_0 - (I_1 / I_0)^2

This can be done using the following code::

    import vegas
    import math
    import gvar as gv

    def f(x):
        dx2 = 0.0
        for d in range(4):
            dx2 += (x[d] - 0.5) ** 2
        f = math.exp(-200 * dx2)
        return [f, f * x[0], f * x[0] ** 2]

    def f_avg(x):
        return sum(f(x)) / 3.

    integ = vegas.Integrator(4 * [[0, 1]])

    # adapt grid to f_avg
    training = integ(f_avg, nitn=10, neval=1000)
    print('Adaptation:\n')
    print(training.summary())

    # evaluate multi-integrands
    print('\nFinal Results:\n')
    result = integ.multi(f, nitn=20)
    print('I[0] =', result[0], '  I[1] =', result[1], '  I[2] =', result[2], '\n')
    print('<x> =', result[1] / result[0])
    print(
        'sigma_x**2 = <x**2> - <x>**2 =', 
        result[2] / result[0] - (result[1] / result[0]) ** 2
        )
    print('\ncorrelation matrix:\n', gv.evalcorr(result))

Here we first train |vegas| on the the average of the three integrands;
any one of the integrands could have used by itself, but the average costs about
the same to compute. 
We then use |vegas| to generate ``nitn=20`` values for each integral.
|vegas| computes the averages of these 20 values, 
as well as the covariance matrix for the resulting estimates. Here
``result`` is an array containing two objects representing Gaussian
random variables --- type :class:`gvar.GVar` from the ``lsqfit`` package
which must be installed in order to use ``Integrator.multi()``. These 
objects encode information about the mean value (``result[i].mean``) 
and standard deviation (``result[i].sdev``) for the estimates for each
integral. They also encode information about correlations between 
different Gaussian variables.

The code produces the following output:

.. literalinclude:: eg3a.out

The estimates for the individual integrals are separately accurate to 
about ±0.4%, 
but the estimate for :math:`\langle x \rangle = I_1/I_0` is accurate to ±0.06%.
This is almost an order
of magnitude (9.5x) more accurate than we would obtain absent correlations. 
The correlation matrix shows that there is 99% correlation between the
statistical fluctuations in estimates for :math:`I_0` and :math:`I_1`,
and so the bulk of these fluctuations cancel in the ratio. 
The estimate for the variance :math:`\sigma^2_x`
is almost two orders of magnitude (92x) more accurate than we would 
have obtained had the integrals been evaluated separately. Both estimates
are correct to within the quoted errors. 

|vegas| estimates the covariance matrix from the variations in the integral
estimates between ``nitn`` different iterations. Thus the error estimates 
(i.e., the standard deviations) have fractional statistical errors
of order ``1/sqrt(2 * nitn)``, provided, of course, 
that ``neval`` is large enough so that
the estimates are Gaussian. Using ``nitn=10`` or ``20`` means that the error 
estimates are accurate to 15--20%, which is more than accurate enough for most 
applications.

As always, |vegas| is faster if the integrand is vectorized. The above 
example could have been written ::

    import vegas
    import math
    import numpy as np
    import gvar as gv

    class f(vegas.VecIntegrand):
        def __call__(self, x):
            x = np.asarray(x)
            fv = np.empty((x.shape[0], 3), float)
            dx2 = 0.0
            for d in range(4):
                dx2 += (x[:, d] - 0.5) ** 2
            fv[:, 0] = np.exp(-200. * dx2)
            fv[:, 1] = fv[:, 0] * x[:, 0]
            fv[:, 2] = fv[:, 0] * x[:, 0] ** 2
            return fv

    class f_avg(vegas.VecIntegrand):
        def __init__(self):
            self.fcn = f()
        def __call__(self, x, fs, nx):
            x = np.asarray(x)[:nx, :]
            fs = np.asarray(fs)[:nx]
            fv = self.fcn(x)
            fs[:] = (fv[:, 0] + fv[:, 1] + fv[:, 2]) / 3.

    integ = vegas.Integrator(4 * [[0, 1]])

    # adapt grid to f_avg
    training = integ(f_avg(), nitn=10, neval=1000)
    print('Adaptation:\n')
    print(training.summary())

    # evaluate multi-integrands
    print('\nFinal Results:\n')
    result = integ.multi(f(), nitn=20)
    print('I[0] =', result[0], '  I[1] =', result[1], '  I[2] =', result[2], '\n')
    print('<x> =', result[1] / result[0])
    print(
        'sigma_x**2 = <x**2> - <x>**2 =', 
        result[2] / result[0] - (result[1] / result[0]) ** 2
        )
    print('\ncorrelation matrix:\n', gv.evalcorr(result))


with identical results but faster execution time. 
Even better would be a Cython 
version. Here the vector integrand returns integrand values in 
arrays ``f[i, s]`` where ``i`` labels different integration points
while ``s=0, 1`` labels the different integrands. Multi-dimensional
integrands, with ``f[i, s1, s2, ...]``, are also allowed.

|vegas| as a Random Number Generator
-------------------------------------
Having adapted to an integrand, |vegas| generates random points
in the integration volume from a distribution that was optimized
for the integrand. It is possible to access integration points
generated by |vegas|, together with
the weights they carry in an integral, using the iterators
:meth:`vegas.Iterator.random` and :meth:`vegas.Iterator.random_vec`.
For example, ::

    integ = vegas.iterator(...)
    ...
    integ(f_training)
    ...
    for itn in range(10):
        # estimate integral of f()
        integral = 0.0
        for x, wgt in integ.random_vec():
            for i in range(x.shape[0]):
                integral += f(x[i, :]) * wgt[i]
        results.append(integral)

generates 10 Monte Carlo estimates of the integral of function ``f(...)``
using an integrator trained on function ``f_training(...)``. The integration
points ``x[i, d]`` are generated from a distribution optimized for ``f_training``,
which should be similar to ``f``. This low-level
access is useful for implementing analyses like that in the previous section
(:meth:`vegas.Integrator.multi`).


Implementation Notes
---------------------
This implementation relies upon Cython for its speed and
numpy for vector processing. It also uses matplotlib
for graphics, but graphics is optional. 

|vegas| also uses the :mod:`gvar` module from the :mod:`lsqfit` 
package if that package is installed (``pip install lsqfit``).
Integration results are returned as objects of type 
:class:`gvar.GVar`, which is a class representing Gaussian
random variables (i.e., something with a mean and standard 
deviation). These objects can be combined with numbers and 
with each other in arbitrary arithmetic expressions to 
get new :class:`gvar.GVar`\s with the correct standard 
deviations (and properly correlated with other 
:class:`gvar.GVar`\s --- that is the tricky part). 

If :mod:`lsqfit` is not installed, |vegas| uses a limited substitute
that supports arithmetic between :class:`gvar.GVar`\s
and numbers, but not between :class:`gvar.GVar`\s and other
:class:`gvar.GVar`\s. It also supports ``log``, ``sqrt`` 
and ``exp`` of :class:`gvar.GVar`\s, but not trig functions 
--- for these install the lsqfit package. Also the  
multi-integrand method :meth:`vegas.Integrator.multi` 
requires the :class:`gvar.GVar` from ``lsqfit``; the substitute
doesn't work for that method.

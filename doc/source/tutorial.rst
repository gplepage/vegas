Overview and Tutorial
=======================================

.. moduleauthor:: G. Peter Lepage <g.p.lepage@cornell.edu>

.. |Integrator| replace:: :class:`vegas.Integrator`
.. |AdaptiveMap| replace:: :class:`vegas.AdaptiveMap`
.. |vegas| replace:: :mod:`vegas`
.. |WAvg| replace:: :class:`lsqfit.WAvg`
.. |chi2| replace:: :math:`\chi^2`

Introduction
-------------

Class :class:`vegas.Integrator` gives Monte Carlo estimates of arbitrary
(square-integrable) multidimensional integrals using the *vegas* algorithm (G.
P. Lepage, J. Comput. Phys. 27 (1978) 192).  The algorithm has two components.
First an automatic transformation  is applied to to the integration variables
in an attempt to flatten the integrand. Then a Monte Carlo estimate of the
integral is made using the  transformed variables. Flattening the integrand
makes the integral easier and improves the estimate.  The transformation
applied to the integration variables has a restricted form, and is optimized
over several iterations of the algorithm: information about the integrand that
is collected during one iteration is used to  improve the transformation used
in the next iteration.

Monte Carlo integration makes few assumptions about the integrand  beyond
square-integrability --- it needn't be analytic nor even continuous. This
makes Monte Carlo integation unusually robust. It also makes it well suited
for adaptive integration. Adaptive strategies are essential for
multidimensional integration, especially in high dimensions, because
multidimensional space is large, with  lots of corners. For example, 90% of
the 1-dimensional integral 

.. math::

    \int_0^1 dx\,\mathrm{e}^{- 100 (x-0.5)^2}

comes from about 23% of the integration volume. In 20 dimensions,
90% of the analogous integral, 

.. math::

    \int_0^1dx_1\cdots\int_0^1 dx_{20} 
    \,\,\mathrm{e}^{- 100 \sum_{\mu}(x_\mu-0.5)^2},

comes from only 10\ :sup:`-8` of the total integration volume. Non-adaptive
strategies would have a very hard time noticing that there was a peak at all.
*vegas* has no trouble with this integral.

Monte Carlo integration also provides efficient and reliable methods for
estimating the uncertainty in its results. In particular, each Monte Carlo
estimate of an integral is a Gaussian random number, from a distribution
whose mean is the correct value of the integral, provided the number of
integrand samples is sufficiently large. In practive one generates multiple
estimates in order to verify that the number of samples is sufficiently large
to guarantee Gaussian behavior. Error analysis is straightforward if the
integral estimates are Gaussian.

The *vegas* algorithm has been in use for decades and implementations are
available in may programming languages, including Fortran (the original
version), C and C++. The algorithm used here is significantly improved over
the original implementation, and that used in most other implementations.  The
module is written in cython, so it is almost as fast as optimized Fortran or
C, particularly when the integrand is also coded in cython (or some other
compiled language) --- see below.

Basic Integrals
----------------
Here we illustrate |vegas| by estimating the integral

.. math::

    C\int_{-1}^1 dx_0 \int_0^1 dx_1 \int_0^1 dx_2 \int_0^1 dx_3
    \,\,\mathrm{e}^{- 100 \sum_{\mu}(x_\mu-0.5)^2},

where constant :math:`C` is chosen so that the exact value is 1. 
The following code shows how this can be done::

    import vegas
    import math

    def f(x): 
        dx2 = 0 
        for d in range(4): 
            dx2 += (x[d] - 0.5) ** 2
    return math.exp(-dx2 * 100.) * 1013.2118364296088

    integ = vegas.Integrator(
        [[-1., 1.], [0., 1.], [0., 1.], [0., 1.]],
        analyzer=vegas.reporter(),
        )

    result = integ(f, nitn=10, neval=1000)
    print('result = %s    Q = %.2f' % (result, result.Q))

First we define the integrand ``f(x)`` where ``x`` specifies a  point in the
4-dimensional space. We then create an  integrator, ``integ``, which is an
integration operator  that can be applied to any 4-dimensional function. It is
where we specify the integration volume. The ``analyzer=vegas.reporter()`` is
optional; it causes the integrator to print out intermediate  results as it
evaluates the integral. Finally we apply ``integ`` to our integrand ``f(x)``,
telling the integrator to estimate  the integral using ``nitn=10`` iterations
of the |vegas| algorithm, each of which uses no more than ``neval=1000``
evaluations of the integrand. Each iteration produces an independent
estimate of the integral. The final estimate is the weighted average of
the results from all 10 iterations, and is returned by ``integ(f ...)``.

This code produces the following output:

.. literalinclude:: eg1a.out

There are several things worth noting here:

    **Adaptation:** Integration estimates are shown for each of the 10 iterations,
    giving both the estimate from just that iteration, and the weighted
    average of results from all iterations up to that point. The
    estimates from the first two iterations are not accurate at
    all, with errors equal to 30--90% of the final result. 
    ``integ`` initially has no information about the integrand
    and so does a relatively poor job of estimating the integral.
    The integrand has a large narrow peak in the center of the 
    integration volume. Most of ``integ``'s integrand samples 
    miss the peak in early iterations, but it
    uses information from the samples in one iteration
    to remap the integration variables for subsequent iterations,
    concentrating sampling where the function is largest. 
    As a result, the per iteration error
    is reduced to 4% by the fifth iteration, and below 2% by
    the end --- an improvement by almost a factor of 50 from 
    the start.

    **Weighted Average:** The final result, 0.9991 ± 0.0092, is obtained from a weighted
    average of the separate results from each iteration. The results from
    individual iterations are random numbers whose distribution is 
    Gaussian, about a mean equal to the integral's value, provided
    the number of evaluations per iteration (``neval``) is  sufficiently
    large. The weighted average :math:`\overline I`  minimizes

    .. math::

      \chi^2 \,\equiv\, \sum_i \frac{(I_i - \overline I)^2}{\sigma_{i}}

    where :math:`I_i \pm \sigma_{i}` are the estimates from the 
    individual iterations. If the :math:`I_i` are Gaussian, 
    :math:`\chi^2` should be of order the number of degrees of 
    freedom, which here is the number of iterations minus 1. 
    The error estimates are not reliable if :math:`\chi^2` is
    much larger than the number of iterations. This quantified
    by the *Q* or *p-value* of the weighted average
    which is the probability that a
    larger :math:`\chi^2` could result from random (Gaussian)
    fluctuations. A very small *Q* (less than 0.05-0.1) indicates
    that the :math:`\chi^2` is too large to be accounted for by
    statistical fluctuations --- that is, the estimates of the integral
    from different iterations do not agree with each other to 
    within errors. This means that ``neval`` is not sufficiently
    large to guarantee Gaussian behavior, and must be increased
    if the error estimates are to be trusted:


    ``integ(f)`` returns a weighted-average object,
    of type :class:`RunningWAvg`, that has the following 
    attributes:

      ``result.mean`` --- weighted average of all estimates of the integral;
      
      ``result.sdev`` --- standard deviation of the weighted average;
      
      ``result.chi2`` --- :math:`\chi^2` of the weighted average;

      ``result.dof`` --- nuumber of degrees of freedom;

      ``result.Q`` --- *Q* or *p-value* of the weighted average's |chi2|;

      ``result.itn_results`` --- list of the integral estimates 
      from each iteration.

    In this example the final *Q* is 0.22, indicating that the
    :math:`\chi^2` for this average is not particularly unlikely.

    **More Precision:** For realistic problems, 
    the cost of a |vegas| integral is 
    usually dominated by the cost of evaluating the integrand
    at the Monte Carlo sample points. The number of integrand
    evaluations per iteration varies from iteration to iteration,
    here between 486 and 952. Typically |vegas| needs more
    integration points in early iterations, before it has fully
    adapted to the integrand.

    Although precision can be increased by increasing either
    the number of iterations (``nitn``) or the number of
    integrand evaluations per iteration (``neval``), it is
    generally far better to increase ``neval``. For example,
    adding the following lines to the code above ::

      integ.set(analyzer=None)      # turn off verbose output

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
    point where vegas has fully adapted. 

    It is also generally useful to compare results from estimates 
    using different values of ``neval``, differing by factors of
    4--10 say. Insofar as the two results agree within errors,
    it is unlikely that non-Gaussian artifacts from small
    ``neval``\s are important. (These artifacts typically vanish
    like ``1/neval``, which is faster than the statistical 
    errors vanish; so the latter ultimately dominate at large
    ``neval``.)

    **Early Iterations:** The early iterations, before |vegas| has adapted, are quite 
    crude. With very peaky integrands, these can be far from 
    the correct answer with highly unreliable errors. For 
    example, the integral above becomes much more 
    difficult (because the peak is 2\ :sup:`4` times harder to
    find) if we double the length of each side of the 
    integration volume by redefining ``integ`` as::

      integ = vegas.Integrator(
        [[-2., 2.], [0, 2.], [0, 2.], [0., 2.]],
        analyzer=reporter(),
        )

    Then the code above gives:

    .. literalinclude:: eg1c.out

    |vegas| misses the peak completely in the first few iterations,
    giving estimates that are wrong (by 670,000 standard deviations!).
    Some of its samples hit the peak's shoulders, so ``integ`` is 
    eventually able to find it (by iterations 7--8), but 
    the integrand estimates are wildly non-Gaussian before that
    point. This results in a non-sensical final result, as 
    indicated by the ``Q = 0.00``. 

    It is common practice in using |vegas| to discard 
    estimates from the first several iterations, before the 
    algorithm has adapted, in order to avoid ruining the 
    final result in this way. This is done by replacing the 
    single call to ``integ(f ...)`` in the original code 
    with two calls::

      # step 1 -- adapt to f; discard results
      integ(f, nitn=7, neval=1000)

      # step 2 -- integ has adapted to f; keep results
      result = integ(f, nitn=10, neval=1000)
      print('result = %s    Q = %.2f' % (result, result.Q))

    The results from the second step are properly adapted from
    the start, and the final result is good:

    .. literalinclude:: eg1d.out

    **Other Integrands:** Once ``integ`` is trained on ``f(x)``, it can be usefully applied
    to other functions with similar structure. For example, adding
    the following at the end of the original code, ::

      def g(x):
        return x[0] * f(x)

      result = integ(g, nitn=10, neval=1000)

    gives the following new output:

    .. literalinclude:: eg1e.out

    The grid is almost optimal for ``g(x)`` from the start
    because ``g(x)`` peaks in the same region as ``f(x)``.
    The exact value for this integral is 0.5.

    Note that |Integrator|\s can be saved in files and reloaded later using
    Python's :mod:`pickle` module: for example, 
    ``pickle.dump(integ, openfile)`` saves integrator ``integ``, and 
    ``integ = pickle.load(openfile)`` reloads it. The is useful for costly
    integrations that might need to be reanalyzed later since the integrator
    remembers the variable transformations made to minimize errors, and
    so need not be readapted to the integrand when used later.


Faster Integrands
-------------------------
Realistic applications of multi-dimensional integration can 
require millions or hundreds of millions of evaluations of 
the integrand. 

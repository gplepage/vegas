:mod:`vegas` Module
==================================================================

.. |Integrator| replace:: :class:`vegas.Integrator`
.. |AdaptiveMap| replace:: :class:`vegas.AdaptiveMap`
.. |vegas| replace:: :mod:`vegas`
.. |RAvg| replace:: :class:`vegas.RAvg`
.. |chi2| replace:: :math:`\chi^2`
.. |x| replace:: x
.. |y| replace:: y
.. |GVar| replace:: :class:`gvar.GVar`

.. moduleauthor:: G. Peter Lepage <g.p.lepage@cornell.edu>

.. module:: vegas
     :synopsis: Adaptive multidimensional Monte Carlo integration

Introduction
----------------
The key Python objects supported by the |vegas| module are:

   * |Integrator| --- an object describing a multidimensional integration
     operator. Such objects contain information about the integration volume,
     and also about optimal remappings of the integration variables based
     upon the last integral evaluated using the object.

   * |AdaptiveMap| --- an object describing the remappings used by |vegas|.

   * |RAvg| --- an object describing the result of a |vegas| integration.
     |vegas| returns the weighted average of the integral estimates
     from each |vegas| iteration as an object of class |RAvg|. These are
     Gaussian random variables --- that is, they have a
     mean and a standard deviation --- but also contain information about the
     iterations |vegas| used in generating the result.

   * :class:`vegas.RAvgArray` --- an array version of |RAvg| used when
     the integrand is array-valued.

   * :class:`vegas.RAvgDict` --- a dictionary version of |RAvg| used when
     the integrand is dictionary-valued.

   * :class:`vegas.PDFIntegrator` --- a specialized integrator for evaluating
     Gaussian expectation values.

These are described in detail below.


Integrator Objects
----------------------------
The central component of the |vegas| package is the integrator class:

.. autoclass:: vegas.Integrator

    |Integrator| objects have attributes for each of these parameters.
    In addition they have the following methods:

  Methods:
    .. automethod:: __call__(fcn, **kargs)

    .. automethod:: set(ka={}, **kargs)

    .. automethod:: settings(ngrid=0)

    .. automethod:: random(yield_hcube=False, yield_y=False)

    .. automethod:: random_batch(yield_hcube=False, yield_y=False)


AdaptiveMap Objects
---------------------
|vegas|’s remapping of the integration variables is handled
by a :class:`vegas.AdaptiveMap` object, which maps the original
integration variables |x| into new variables |y| in a unit hypercube.
Each direction has its own map specified by a grid in |x| space:

    .. math::

        x_0 &= a \\
        x_1 &= x_0 + \Delta x_0 \\
        x_2 &= x_1 + \Delta x_1 \\
        \cdots \\
        x_N &= x_{N-1} + \Delta x_{N-1} = b

where :math:`a` and :math:`b` are the limits of integration.
The grid specifies the transformation function at the points
:math:`y=i/N` for :math:`i=0,1\ldots N`:

    .. math::

        x(y\!=\!i/N) = x_i

Linear interpolation is used between those points. The Jacobian
for this transformation is:

    .. math::

        J(y) = J_i = N \Delta x_i

|vegas| adjusts the increments sizes to optimize its Monte Carlo
estimates of the integral. This involves training the grid. To
illustrate how this is done with |AdaptiveMap|\s consider a simple
two dimensional integral over a unit hypercube with integrand::

   def f(x):
      return x[0] * x[1] ** 2

We want to create a grid that optimizes uniform Monte Carlo estimates
of the integral in |y| space. We do this by sampling the integrand
at a large number ``ny`` of random points ``y[j, d]``, where ``j=0...ny-1``
and ``d=0,1``, uniformly distributed throughout the integration
volume in |y| space. These samples be used to train the grid
using the following code::

   import vegas
   import numpy as np

   def f(x):
      return x[0] * x[1] ** 2

   m = vegas.AdaptiveMap([[0, 1], [0, 1]], ninc=5)

   ny = 1000
   y = np.random.uniform(0., 1., (ny, 2))  # 1000 random y's

   x = np.empty(y.shape, float)            # work space
   jac = np.empty(y.shape[0], float)
   f2 = np.empty(y.shape[0], float)

   print('intial grid:')
   print(m.settings())

   for itn in range(5):                    # 5 iterations to adapt
      m.map(y, x, jac)                     # compute x's and jac

      for j in range(ny):                  # compute training data
         f2[j] = (jac[j] * f(x[j])) ** 2

      m.add_training_data(y, f2)           # adapt
      m.adapt(alpha=1.5)

      print('iteration %d:' % itn)
      print(m.settings())

In each of the 5 iterations, the |AdaptiveMap| adjusts the
map, making increments smaller where ``f2`` is larger and
larger where ``f2`` is smaller.
The map converges after only 2 or 3 iterations, as
is clear from the output:

.. literalinclude:: eg2a.out

The grid increments along direction 0 shrink at larger
values ``x[0]``, varying as ``1/x[0]``. Along direction 1
the increments shrink more quickly varying like ``1/x[1]**2``.

|vegas| samples the integrand in order to estimate the integral.
It uses those same samples to train its |AdaptiveMap| in this
fashion, for use in subsequent iterations of the algorithm.

.. autoclass:: vegas.AdaptiveMap

  Attributes and methods:
   .. autoattribute:: dim

   .. autoattribute:: ninc

         ``ninc[d]`` is the number increments in direction ``d``.

   .. attribute:: grid

      The nodes of the grid defining the maps are ``self.grid[d, i]``
      where ``d=0...`` specifies the direction and ``i=0...self.ninc``
      the node.

   .. attribute:: inc

      The increment widths of the grid::

          self.inc[d, i] = self.grid[d, i + 1] - self.grid[d, i]

   .. automethod:: adapt(alpha=0.0, ninc=None)

   .. automethod:: add_training_data(y, f, ny=-1)

   .. automethod:: adapt_to_samples(x, f, nitn=5, alpha=1.0)

   .. automethod:: __call__(y)

   .. automethod:: jac(y)

   .. automethod:: make_uniform(ninc=None)

   .. automethod:: map(y, x, jac, ny=-1)

   .. automethod:: invmap(x, y, jac, nx=-1)

   .. automethod:: show_grid(ngrid=40, shrink=False)

   .. automethod:: settings(ngrid=5)

   .. automethod:: extract_grid()

   .. automethod:: clear()


PDFIntegrator Objects
-----------------------------------------------------------
Expectation values using probability density functions defined by
collections of Gaussian random variables (see :mod:`gvar`)
can be evaluated using the following
specialized integrator:

.. autoclass:: vegas.PDFIntegrator(g, limit=1e15, scale=1., svdcut=1e-15)

  Methods:
    .. automethod:: __call__(f, nopdf=False, **kargs)


Other Objects and Functions
----------------------------
.. autoclass:: vegas.RAvg

  Attributes and methods:
   .. attribute:: mean

      The mean value of the weighted average.

   .. attribute:: sdev

      The standard deviation of the weighted average.

   .. autoattribute:: chi2

   .. autoattribute:: dof

   .. autoattribute:: Q

   .. attribute:: itn_results

      A list of the results from each iteration.

   .. attribute:: sum_neval 

      Total number of integrand evaluations used in all iterations.

   .. automethod:: add(g)

   .. automethod:: summary(weighted=None)

.. autoclass:: vegas.RAvgArray

  Attributes and methods:
   .. autoattribute:: chi2

   .. autoattribute:: dof

   .. autoattribute:: Q

   .. attribute:: itn_results

      A list of the results from each iteration.

   .. attribute:: sum_neval 

      Total number of integrand evaluations used in all iterations.

   .. automethod:: add(g)

   .. automethod:: summary(extended=False, weighted=None)

.. autoclass:: vegas.RAvgDict

  Attributes and methods:
   .. autoattribute:: chi2

   .. autoattribute:: dof

   .. autoattribute:: Q

   .. attribute:: itn_results

      A list of the results from each iteration.

   .. attribute:: sum_neval 

      Total number of integrand evaluations used in all iterations.

   .. automethod:: add(g)

   .. automethod:: summary(extended=False, weighted=None)

.. autofunction:: vegas.batchintegrand

.. autoclass:: vegas.BatchIntegrand



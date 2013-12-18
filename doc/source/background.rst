How :mod:`vegas` Works
========================

.. moduleauthor:: G. Peter Lepage <g.p.lepage@cornell.edu>

.. |Integrator| replace:: :class:`vegas.Integrator`
.. |AdaptiveMap| replace:: :class:`vegas.AdaptiveMap`
.. |vegas| replace:: :mod:`vegas`
.. |WAvg| replace:: :class:`lsqfit.WAvg`
.. |chi2| replace:: :math:`\chi^2`
.. |x| replace:: x 
.. |y| replace:: y 


The |vegas| Grid: Importance Sampling
---------------------------------------
The most important adaptive strategy |vegas| uses is 
its remapping of the integration variables in each 
direction, before it makes Monte Carlo estimates of the integral.
In one dimension, for example, |vegas| converts

.. math::

    I = \int_a^b dx\,f(x)

into 

.. math::
    
    I = \int_0^1 dy\, J(y)\,f(x(y))

and then does a Monte Carlo estimate in |y| space.
The transformation function :math:`x(y)` is chosen so that 
Jacobian :math:`J(y)\propto 1/|f(x)|`. This choice minimizes
the statistical errors of the |y|-space Monte Carlo estimate 
of integral: it flattens the integrand, removing peaks by 
spreading them out in |y| space. 

|vegas| implements this transformation using a grid in |x| space:

    .. math::

        x_0 &= a \\
        x_1 &= x_0 + \Delta x_0 \\
        x_2 &= x_1 + \Delta x_1 \\
        \cdots \\
        x_N &= x_{N-1} + \Delta x_{N-1} = b

The grid specifies the transformation function at the points 
:math:`y=i/N` for :math:`i=0,1\ldots N`:

    .. math::

        x(y\!=\!i/N) = x_i

Linear interpolation is used between those points. 

The Jacobian for this transformation function is piecewise constant:

    .. math:: 

        J(y) = N \Delta x_i

for :math:`i/N < y < (i+1)/N`. Consequently the optimal grid is one 
where the grid increments :math:`\Delta x_i` are small in regions
where :math:`|f(x)|` is large. These small increments map into
proportionally larger increments in |y| space. 
This causes the Monte Carlo
integration to focus its sampling of the integrand in regions where the 
integrand
is large --- a standard Monte Carlo technique 
called "importance sampling."

|vegas| typically starts with no knowledge of the integrand and 
so starts with a uniform |x| grid. As it samples the integrand
it gathers information about where the peaks are, and refines
its choice of :math:`\Delta x_i`\s for use
in subsequent iterations. The grid converges after several iterations
to the optimal grid.

This analysis generalizes easily to multi-dimensional integrals. 
|vegas| applies a similar transformation in each direction, and 
the grid increments along an axis 
are made smaller in regions where the 
projection of the integral onto that axis is larger. For example,
the optimal grid for the four-dimensional Gaussian integral
in the previous section looks like:

.. image:: eg1a-plt1.*
   :width: 80%

.. image:: eg1a-plt2.*
   :width: 80%

These plots were obtained by including the line ::

    integ.map.plot_grid(30)

in the integration code after the integration is finished.
It causes :mod:`matplotlib` (if it is installed) to create 
images showing 30 nodes (out of the 99 actually used) of 
the grid in each direction. Obviously |vegas| is focusing
its resources on the region around ``x = [0.5, 0.5, 0.5, 0.5]``.


Adaptive Stratified Sampling
-------------------------------

A limitation of |vegas|’s remapping strategy becomes obvious if we look
at the grid for the following integral, which has two Gaussians
arranged along the diagonal of the hypercube::

    def f2(x): 
        dx2 = 0 
        for i in range(4): 
            dx2 += (x[i] - 1/3.) ** 2
        ans = math.exp(-dx2 * 100.) * 1013.2167575422921535
        dx2 = 0 
        for i in range(4): 
            dx2 += (x[i] - 2/3.) ** 2
        ans += math.exp(-dx2 * 100.) * 1013.2167575422921535
        return ans / 2.

    integ = vegas.Integrator(4 * [[0, 1]])

    integ(f2, nitn=10, neval=4e4)
    result = integ(f2, nitn=30, neval=4e4)
    print('result = %s    Q = %.2f' % (result, result.Q))

    integ.map.plot_grid(70)

This code gives the following grid, now showing 70 nodes
in each direction:

.. image:: eg1h-plt1.png
    :width: 80%

The grid shows that |vegas| is concentrating on the regions
around ``x=[0.33, 0.33, 0.33, 0.33]`` and 
``x=[0.67, 0.67, 0.67, 0.67]``, where the peaks are.
Unfortunately it is also concentrating on regions around
points like ``x=[0.67, 0.33, 0.33, 0.33]`` where the integrand
is very close to zero. There are 14 such phantom peaks
that |vegas|’s new integration variables emphasize, 
in addition to the 2 regions
where the integrand actually is large. This grid gives
much better results
than using a uniform grid, but it obviously 
wastes integration resources.
It is a consequence
of the fact that |vegas| remaps the integration variables in
each direction separately. Projected on the ``x[0]`` axis, for example,
this integrand appears to have two peaks and so |vegas| will
focus on both regions of ``x[0]``, independently of what it does
along the ``x[1]`` axis.

|vegas| uses axis-oriented remappings because other 
alternatives are much more complicated and expensive; and |vegas|’s
principal adaptive strategy has proven very effective in lots 
of realistic applications. 

An axis-oriented
strategy will always have difficulty adapting to structures that
lie along diagonals of the integration hypercube. To address such problems,
this new version of |vegas| introduces a second adaptive strategy,
based upon another standard Monte Carlo technique called "stratified
sampling." |vegas| divides the d-dimensional 
|y|-space volume into hypercubes using
a uniform |y|-space grid with M stratifications on each 
axis. It estimates
the integral by doing a separate Monte Carlo integration in each of 
the M\ :sup:`d` hypercubes, and adding the results together to provide an estimate
for the integral over the entire integration region.
Typically 
this |y|-space grid is much coarser than the |x|-space grid used to 
remap the integration variables. This is because |vegas| needs 
at least two integrand evaluations in each |y|-space hypercube, and
so must keep the number of hypercubes M\ :sup:`d` smaller than ``neval/2``. 
This restricts M when d is large.

Older versions of |vegas| also divide |y|-space into hypercubes and 
do Monte Carlo estimates in the separate hypercubes. These versions, however,
use the same number of integrand evaluations in each hypercube. 
In the new version, |vegas| adjusts the number of evaluations used 
in a hypercube in proportion to the standard deviation of 
the integral estimate from that hypercube --- it concentrates
integration evaluations where the statistical errors are 
largest. In the two-Gaussian example above, for example, 
it shifts
integration evaluations away from the phantom peaks, into
the regions occupied by the real peaks since this is where all
the error comes from.

This new strategy significantly reduces the statistical errors for integrals
with large diagonal structures, like the two-Gaussian integral,
provided ``neval`` is large enough to permit a large number M (more 
than 2 or 3) of
stratifications on each axis. For the two-Gaussain integral, the new adaptive
strategy (i.e., adaptive stratified sampling) reduces statistical 
errors by more than a factor of 3 over what older versions of
|vegas| give. This is a relatively easy integral; 
the difference can be more 
than an order of magnitude for more difficult (and realistic)
integrals.





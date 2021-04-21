How :mod:`vegas` Works
========================

.. moduleauthor:: G. Peter Lepage <g.p.lepage@cornell.edu>

.. |Integrator| replace:: :class:`vegas.Integrator`
.. |AdaptiveMap| replace:: :class:`vegas.AdaptiveMap`
.. |vegas| replace:: :mod:`vegas`
.. |WAvg| replace:: :class:`vegas.RunningWAvg`
.. |chi2| replace:: :math:`\chi^2`
.. |x| replace:: :math:`x`
.. |y| replace:: :math:`y`
.. |S1| replace:: :math:`S^{(1)}`
.. |M| replace:: :math:`M`
.. |sigmaI| replace:: :math:`\sigma_I`
.. |x(y)| replace:: :math:`x(y)`
.. |Ms| replace:: :math:`M_\mathrm{st}`
.. |Msplus| replace:: :math:`M_\mathrm{st}+1`
.. |Msd| replace:: :math:`(M_\mathrm{st}+1)^{D_0} M_\mathrm{st}^{D-D_0}`
.. |d| replace:: :math:`D`
.. |d0| replace:: :math:`D_0`

|vegas| uses two adaptive strategies: importance sampling, and
adaptive stratified sampling. Here we discuss the ideas behind each,
in turn.

.. _importance_sampling:

Importance Sampling
------------------------------------------------
The most important adaptive strategy |vegas| uses is
its remapping of the integration variables in each
direction, before it makes Monte Carlo estimates of the integral.
This is equivalent to a standard Monte Carlo optimization
called "importance sampling."

|vegas| chooses transformations
for each integration variable
that minimize the statistical errors in
Monte Carlo estimates whose integrand
samples are uniformly distributed
in the new variables.
The idea in one-dimension, for
example, is to replace the original integral over |x|,

.. math::

    I = \int_a^b dx\; f(x),

by an equivalent integral over a new variable |y|,

.. math::

    I = \int_0^1 dy\; J(y)\; f(x(y)),

where :math:`J(y)` is the Jacobian of the transformation.
A simple Monte Carlo estimate of the transformed
integral is given by

.. math::

	I \approx S^{(1)} \equiv \frac{1}{M} \sum_y \;J(y)\; f(x(y))

where the sum is over |M| random points
uniformly distributed between 0 and 1.

The estimate |S1| is a itself a random number from a distribution
whose mean is the exact integral and whose variance is:

.. math::

	\sigma_I^2 &= \frac{1}{M}\left(
	\int_0^1 dy\; J^2(y) \; f^2(x(y)) - I^2
	\right) \\
	&= \frac{1}{M}\left(
	\int_a^b dx \;J(y(x))\; f^2(x) - I^2
	\right)

The standard deviation |sigmaI| is an estimate of the possible
error in the Monte Carlo estimate.
A straightforward variational calculation, constrained by

.. math::

	\int_a^b \frac{dx}{J(y(x))} = \int_0^1 dy = 1,

shows that |sigmaI| is minimized if

.. math::

	J(y(x)) = \frac{\int_a^b dx\;|f(x)|}{|f(x)|}.

Such transformations greatly reduce the standard deviation when the
integrand has high peaks. Since

.. math::

	1/J = \frac{dy}{dx} \propto |f(x)|,

the regions in |x| space where :math:`|f(x)|` is large are
stretched out in |y| space. Consequently, a uniform Monte Carlo in |y| space
places more samples in the peak regions than it would
if were we integrating in |x| space --- its samples are concentrated
in the most important regions, which is why this is called "importance
sampling." The product :math:`J(y)\;f(x(y))` has no peaks when
the transformation is optimal.

The distribution of the Monte Carlo estimates |S1| becomes
Gaussian in the limit of large |M|. Non-Gaussian corrections
vanish like :math:`1/M`. For example, it is easy to show that

.. math::

	\langle (S^{(1)} - I) ^ 4 \rangle
	=
	3\sigma_I^4\left( 1 - \frac{1}{M}\right)
	+ \frac{1}{M^3} \int_0^1 dy \;
	(J(y)\;f(x(y)) - I)^4

This moment would equal :math:`3\sigma_I^4`, which falls like :math:`1/M^2`,
if the distribution was Gaussian. The corrections to the Gaussian result
fall as :math:`1/M^3` and so become negligible at large :math:`M`.
These results assume
that :math:`(J(y)\:f(x(y)))^n` is integrable for all :math:`n`,
which need not be the case
if :math:`f(x)` has (integrable) singularities.

The |vegas| Map
--------------------
|vegas| implements the transformation of an integration variable
|x| into a new variable |y| using a grid in |x| space:

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

        J(y) = J_i = N \Delta x_i

for :math:`i/N < y < (i+1)/N`.

The variance for a Monte Carlo estimate using this transformation
becomes

.. math::

	\sigma_I^2 = \frac{1}{M}\left(
	\sum_i J_i \int_{x_i}^{x_{i+1}} dx \; f^2(x) - I^2
	\right)

Treating the :math:`J_i` as independent variables, with the
constraint

.. math::

	\sum_i \frac{\Delta x_i}{J_i} = \sum_i \Delta y_i = 1,

it is trivial to show that the standard deviation is minimized
when

.. math::

	\frac{J_i^2}{\Delta x_i}
	\int_{x_i}^{x_{i+1}} dx \; f^2(x)
	= N^2 \Delta x_i \int_{x_i}^{x_{i+1}} dx \; f^2(x)
	\; = \; \mbox{constant}

for all :math:`i`.

|vegas| adjusts the grid until this last condition is
satisfied.  As a result grid increments :math:`\Delta x_i` are
small in regions where :math:`|f(x)|` is large.
|vegas| typically has no knowledge of the integrand initially, and
so starts with a uniform |x| grid. As it samples the integrand
it also estimates the integrals

.. math::

	\int_{x_i}^{x_{i+1}} dx \; f^2(x),

and use this information to refine
its choice of :math:`\Delta x_i`\s, bringing them closer to their optimal
values, for use
in subsequent iterations. The grid usually converges,
after several iterations,
to the optimal grid.

This analysis generalizes easily to multi-dimensional integrals.
|vegas| applies a similar transformation in each direction, and
the grid increments along an axis
are made smaller in regions where the
projection of the integral onto that axis is larger. For example,
the optimal grid for the four-dimensional Gaussian integral
in the section on :ref:`basic_integrals` looks like:

.. image:: eg1a-plt1.*
   :width: 80%

.. image:: eg1a-plt2.*
   :width: 80%

These grids transform into uniformly-spaced grids in |y| space.
Consequently a uniform, |y|-space Monte Carlo places the same
number of integrand evaluations, on average, in every rectangle
of these pictures. (The average number is typically much less one
in higher dimensions.) Integrand evaluations are concentrated
in regions where the |x|-space rectangles are small
(and therefore numerous) ---
here in the vicinity of ``x = [0.5, 0.5, 0.5, 0.5]``, where the
peak is.

These plots were obtained by including the line ::

    integ.map.show_grid(30)

in the integration code after the integration is finished.
It causes :mod:`matplotlib` (if it is installed) to create
images showing the locations of 30 nodes
of
the grid in each direction. (The grid uses 99 nodes in all
on each axis, but that is too many to display at low resolution.)

.. _adaptive-stratified-sampling:

Adaptive Stratified Sampling
-------------------------------

A limitation of |vegas|’s remapping strategy becomes obvious if we look
at the grid for the following integral, which has two Gaussians
arranged along the diagonal of the hypercube::

    import vegas
    import math

    def f2(x):
        dx2 = 0
        for d in range(4):
            dx2 += (x[d] - 1/3.) ** 2
        ans = math.exp(-dx2 * 100.) * 1013.2167575422921535
        dx2 = 0
        for d in range(4):
            dx2 += (x[d] - 2/3.) ** 2
        ans += math.exp(-dx2 * 100.) * 1013.2167575422921535
        return ans / 2.

    integ = vegas.Integrator(4 * [[0, 1]])

    integ(f2, nitn=10, neval=4e4)
    result = integ(f2, nitn=30, neval=4e4)
    print('result = %s    Q = %.2f' % (result, result.Q))

    integ.map.show_grid(70)

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
The waste occurs because
|vegas| remaps the integration variables in
each direction separately. Projected on the ``x[0]`` axis, for example,
this integrand appears to have two peaks and so |vegas| will
focus on both regions of ``x[0]``, independently of what it does
along the ``x[1]`` axis.

|vegas| uses axis-oriented remappings because other
alternatives are much more complicated and expensive; and |vegas|’s
principal adaptive strategy has proven very effective in
many realistic applications.

An axis-oriented
strategy will always have difficulty adapting to structures that
lie along diagonals of the integration volume. To address such problems,
the new version of |vegas| introduces a second adaptive strategy,
based upon another standard Monte Carlo technique called "stratified
sampling." |vegas| divides the |d|-dimensional
|y|-space volume into |Msd| hypercubes using
a uniform |y|-space grid with |Ms| or |Msplus| stratifications on each
axis. It estimates
the integral by doing a separate Monte Carlo integration in each of
the hypercubes, and adding the results together to provide an estimate
for the integral over the entire integration region.
Typically
this |y|-space grid is much coarser than the |x|-space grid used to
remap the integration variables. This is because |vegas| needs
at least two integrand evaluations in each |y|-space hypercube, and
so must keep the number of hypercubes |Msd| smaller than ``neval/2``.
This can restrict |Ms| severely when |d| is large.

Older versions of |vegas| also divide |y|-space into hypercubes and
do Monte Carlo estimates in the separate hypercubes. These versions, however,
use the same number of integrand evaluations in each hypercube.
In the new version, |vegas| adjusts the number of evaluations used
in a hypercube in proportion to the standard deviation of
the integrand estimates (in |y| space) from that hypercube.
It uses information about the hypercube's standard deviation in one
iteration to set the number of evaluations for that hypercube
in the next iteration. In this way it concentrates
integrand evaluations where the potential statistical errors are
largest.

In the two-Gaussian example above, for example,
the new |vegas| shifts
integration evaluations away from the phantom peaks, into
the regions occupied by the real peaks since this is where all
the error comes from. This improves |vegas|’s ability to estimate
the contributions from the real peaks and
reduces statistical errors,
provided ``neval`` is large enough to permit a large number  (more
than 2 or 3) |Ms| of
stratifications on each axis. With ``neval=4e4``,
statistical errors for the two-Gaussian
integral are reduced by more than a factor of 3 relative to what older
versions of |vegas| give. This is a relatively easy integral;
the difference can be much larger for more difficult (and realistic)
integrals.





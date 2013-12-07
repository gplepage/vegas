Overview and Tutorial
=======================================

.. moduleauthor:: G. Peter Lepage <g.p.lepage@cornell.edu>

.. |Integrator| replace:: :class:`vegas.Integrator`
.. |AdaptiveMap| replace:: :class:`vegas.AdaptiveMap`
.. |vegas| replace:: :mod:`vegas`
.. |WAvg| replace:: :class:`lsqfit.WAvg`

Introduction
-------------
Class :class:`vegas.Integrator` gives Monte Carlo estimates of 
arbitrary (square-integrable) multidimensional integrals using
the *vegas* algorithm (G. P. Lepage, J. Comput. Phys. 27(1978) 192).
It automatically remaps the integration variables along each direction
to maximize the accuracy of the Monte Carlo estimates. The remapping
is done over several iterations. Monte Carlo estimates of integrals 
are particularly useful because they provide fairly reliable estimates 
of their accuracy, and also non-trivial cross checks on the reliability
of the error estimates.

The *vegas* algorithm has been in use for decades and implementations
are available in may programming languages, including Fortran (the 
original version), C and C++. The algorithm used here is significantly
improved over the original implementation, and that used in most other 
implementations. The module is written in cython, so it is almost as
fast as optimized Fortran or C (within a factor of 1.5 or so), particularly 
when the integrand is coded in cython (or some other compiled language) 
as well---see below.

Basic Integrals
----------------
A simple example that illustrates the standard idiom for 
using :class:`vegas.Integrator` is::

    import vegas
    import math

    def f(x): dx2 = 0 for i in range(4): dx2 += (x[i] - 0.5) ** 2
    return math.exp(-dx2 * 100.) * (100. / math.pi) ** 2


    integ = vegas.Integrator(
        [[-1., 1.], [0., 1.], [0., 1.], [0., 1.]], 
        nitn=10, 
        neval=1000,
        )
    ans = integ(f)
    print '1st estimate of integral =', ans
    ans = integ(f)
    print '2nd estimate of integral =', ans

This code estimates the integral of a narrow Gaussian, centered
at point ``x = [0.5, 0.5, 0.5, 0.5]``, over a four-dimensional 
volume defined by::

        -1 < x[0] < 1
         0 < x[1] < 1
         0 < x[2] < 1
         0 < x[3] < 1

Each time the integrator ``integ`` is applied to a 4d function ``f(x)``, it
generates a Monte Carlo estimate of the integral of that function. Each
estimate is actually  the weighted average of ``nitn=10`` separate estimates,
coming from 10 iterations of the *vegas* algorithm; and each *vegas*
iteration uses about ``neval=1000`` function evaluations. 

The output from the code above, ::

    1st estimate of integral = 1.0028(89)
    2nd estimate of integral = 0.9998(44),

shows that the first estimate is 1.0028 ± 0.0089, while the 
second estimate is 0.9998 ± 0.0044. (The exact value for the 
integral is 1.0.) The second estimate is substantially more 
accurate than the first. This is because ``integ`` initially 
has no knowledge about the structure of ``f(x)``, and so early
iterations of the *vegas* algorithm are less accurate. As the 
integrator proceeds it iteratively remaps the integration variables
in each direction to increase accuracy. |Integrator| object ``integ``
is fully adapted to the function by the time of the second
estimate in the code above, and so that estimate is more accurate.

We can examine the evolution of ``integ``'s results by modifying
its definition to include a ``reporter`` who prints out information
about each *vegas* iteration::

    integ = vegas.Integrator(
        [[-1., 1.], [0., 1.], [0., 1.], [0., 1.]], 
        nitn=10, 
        neval=1000,
        analyzer=vegas.reporter(),
        )

The first call to ``integ`` generates the following output::

	Integrator Status:
	    1000 (max) integrand evaluations in each of 10 iterations
	    integrator mode = adapt_to_integrand
	                      redistribute points across h-cubes
	    number of:  strata/axis = 3  increments/axis = 99
	                h-cubes = 81  evaluations/h-cube = 6 (min)
	                h-cubes/vector = 30
	    damping parameters: alpha = 0.5  beta= 0.75
	    accuracy: relative = 0  absolute accuracy = 0

	    axis 0 covers (-1.0, 1.0)
	    axis 1 covers (0.0, 1.0)
	    axis 2 covers (0.0, 1.0)
	    axis 3 covers (0.0, 1.0)

	    itn  1: 0.56(31)
	 all itn's: 0.56(31)
	    neval = 486  neval/h-cube = (6, 6)
	    chi2/dof = 0.00  Q = 1.0

	    itn  2: 1.30(45)
	 all itn's: 0.79(26)
	    neval = 955  neval/h-cube = (6, 435)
	    chi2/dof = 1.80  Q = 0.2

	    itn  3: 0.934(93)
	 all itn's: 0.918(87)
	    neval = 854  neval/h-cube = (6, 154)
	    chi2/dof = 1.04  Q = 0.4

	    itn  4: 1.044(69)
	 all itn's: 0.996(54)
	    neval = 750  neval/h-cube = (6, 56)
	    chi2/dof = 1.12  Q = 0.3

	    itn  5: 1.005(43)
	 all itn's: 1.002(34)
	    neval = 629  neval/h-cube = (6, 24)
	    chi2/dof = 0.85  Q = 0.5

	    itn  6: 1.016(32)
	 all itn's: 1.009(23)
	    neval = 578  neval/h-cube = (6, 14)
	    chi2/dof = 0.70  Q = 0.6

	    itn  7: 1.028(25)
	 all itn's: 1.018(17)
	    neval = 539  neval/h-cube = (6, 11)
	    chi2/dof = 0.63  Q = 0.7

	    itn  8: 0.978(21)
	 all itn's: 1.002(13)
	    neval = 529  neval/h-cube = (6, 10)
	    chi2/dof = 0.85  Q = 0.5

	    itn  9: 1.012(19)
	 all itn's: 1.006(11)
	    neval = 530  neval/h-cube = (6, 11)
	    chi2/dof = 0.77  Q = 0.6

	    itn 10: 0.997(16)
	 all itn's: 1.0028(89)
	    neval = 529  neval/h-cube = (6, 10)
	    chi2/dof = 0.71  Q = 0.7

Integration estimates are shown here for each of the 10 iterations,
giving both the estimate from just that iteration, together with the
weighted average of results from all iterations up to that  point.
Note how the first two iterations  are not at all accurate, with
uncertainties of order 30--40% of the final results. By the third
iteration the uncertainty has dropped to 9%, and by  the end the
uncertainty from each iteration separately is around 2%. Combining
results from all 10 iterations reduces the uncertainty to less than
1%.

|Integrator| objects like ``integ`` retain information about the
remappings of the integration variables that improve precision
for the last function they analyzed. Consequently when ``integ``
is applied a second time to ``f(x)`` in the code above it has 
already adapted to the function and even the early iterations
are quite accurate::

	Integrator Status:
	    1000 (max) integrand evaluations in each of 10 iterations
	    integrator mode = adapt_to_integrand
	                      redistribute points across h-cubes
	    number of:  strata/axis = 3  increments/axis = 99
	                h-cubes = 81  evaluations/h-cube = 6 (min)
	                h-cubes/vector = 30
	    damping parameters: alpha = 0.5  beta= 0.75
	    accuracy: relative = 0  absolute accuracy = 0

	    axis 0 covers (-1.0, 1.0)
	    axis 1 covers (0.0, 1.0)
	    axis 2 covers (0.0, 1.0)
	    axis 3 covers (0.0, 1.0)

	    itn  1: 1.002(15)
	 all itn's: 1.002(15)
	    neval = 550  neval/h-cube = (6, 11)
	    chi2/dof = 0.00  Q = 1.0

	    itn  2: 0.991(15)
	 all itn's: 0.996(10)
	    neval = 570  neval/h-cube = (6, 12)
	    chi2/dof = 0.26  Q = 0.6

	...

	    itn 10: 0.984(14)
	 all itn's: 0.9998(44)
	    neval = 591  neval/h-cube = (6, 16)
	    chi2/dof = 0.34  Q = 1.0


The final result reported by ``integ(f)`` is the weighted average of
of results from all 10 iterations. Monte Carlo estimates are 
Gaussian random variables provided the number of function evaluations
(``neval``) is large enough. They are characterized by a mean value and a 
standard deviation, representing the best estimate for the value
of the integral and the uncertainty in that estimate. Multiple 
estimates are combined using a weighted average, which yields 
a new Gaussian random variable with a mean of the means and a new
(smaller) standard deviation. Computing the ``chi**2`` of the weighted
average provides an important check on the assumption that ``neval``
is sufficiently large to guarantee Gaussian behavior. The ``chi**2``
divided by the number of degrees of freedom (here 9) should be of 
order one or less. Here ``chi2/dof`` is 0.71, which is fine
(the ``Q`` or *p-value* is 0.7).

``integ(f)`` returns an weighted-average object of type 
:class:`lsqfit.WAvg`. These objects have several attributes::

	ans.mean  ->  average of all estimates of the integral
	ans.sdev  ->  standard deviation of that estimate
	ans.chi2  ->  chi**2 of the weighted average of estimates
	ans.dof   ->  number of degrees of freedom used
	ans.Q     ->  Q or p-value of the average.


Difficult Integrals
------------------------------------
Multidimensional integration for realistic examples is difficult. 
To illustrate some of the problems, consider the integrand from 
the last section but integrated in a volume whose sides
are doubled in length::

	integ = vegas.Integrator(
	    [[-2., 2.],[0., 2.], [0., 2.], [0, 2.]], 
	    nitn=10, 
	    neval=1000,
	    )
    ans = integ(f)
    print '1st integral in larger volume =', ans
    ans = integ(f)
    print '2nd integral in larger volume =', ans

This code gives ::

	1st estimate in larger volume = 0.00103(34)
	2nd estimate in larger volume = 0.9988(57)

where now the first estimate is completely wrong (by ``2938.1`` standard
deviations!). The second estimate is fine. To see what happened with 
first estimate, we again set parameter ``analyzer=vegas.reporter()``
in the constructor for ``integ`` and to obtain the following 
information about the early iterations in the first estimate::

	...

	    itn  1: 0.00034(34)
	 all itn's: 0.00034(34)
	    neval = 591  neval/h-cube = (6, 15)
	    chi2/dof = 0.00  Q = 1.0

	    itn  2: 0.61(21)
	 all itn's: 0.00034(34)
	    neval = 973  neval/h-cube = (6, 493)
	    chi2/dof = 8.68  Q = 0.0

	    itn  3: 0.71(19)
	 all itn's: 0.00034(34)
	    neval = 946  neval/h-cube = (6, 398)
	    chi2/dof = 11.33  Q = 0.0

	    itn  4: 0.93(12)
	 all itn's: 0.00035(34)
	    neval = 863  neval/h-cube = (6, 142)
	    chi2/dof = 28.70  Q = 0.0

	    itn  5: 0.914(53)
	 all itn's: 0.00039(34)
	    neval = 772  neval/h-cube = (6, 50)
    chi2/dof = 96.75  Q = 0.0

    ...

In the first iteration, the integrator has clearly missed the fact
that there is a  giant peak at ``x=[0.5, 0.5, 0.5, 0.5]``. Doubling
the length of each side of the integration volume means that the
fraction of the volume occupied by the peak is 2^4 = 16 times
smaller than it was in the first example. The 591 random samples
of the function in the first iteration were not enough to hit the
peak.  Some of those sample points hit the outer shoulders of the
beak, causing the integrator to concentrate function evaluations in
the general vicinity of the peak in the second iteration. This time it
sees the peak and realizes that it focus still more attention on
that region. It zeros in on the peak over the next few
iterations.

Clearly 591 samples of the function is not enough to make  the
Monte Carlo estimate Gaussian in the first iteration, so  neither the
mean nor the standard deviation is to be trusted for that iteration.
The integrator signals this fact when it reports that the  ``chi**2``
per degree of freedom is much larger than one: by   the tenth
iteration ``chi2/dof = 637``. This large value strongly suggests that
we should ignore the first estimate completely.

For the second estimate, ``ans.chi2/ans.dof`` is 0.61 which 
suggests that that estimate is reliably Gaussian. Consequently
we should feel reasonably confident about the mean and standard
deviation reported by the second estimate.

A common strategy for using the *vegas* algorithm on integrands
with high narrow peaks is to call the integrator twice: a first
time so the integrator can find the peaks and adapt to them, and
a second time to estimate the integral. The mean and standard 
deviation are from the first call are discarded, and the 
``chi**2/dof`` is checked for the second call to verify that 
it is of order one or less. 

Difficult Integrals --- Overly Zealous Adaptation
---------------------------------------------------
Consider the much harder seven-dimensional integral in the 
following example::

    dim = 7         # dimension of integration

    def f(x):       # three narrow Gaussians along the diagonal
        dx2_a = 0.
        dx2_b = 0.
        dx2_c = 0.
        for i in range(dim):
            dx2_a += (x[i] - 0.25) ** 2
            dx2_b += (x[i] - 0.5) ** 2
            dx2_c += (x[i] - 0.75) ** 2
        return (100. / math.pi) ** (dim/2.) / 3. * (
              math.exp(-dx2_a * 100.) 
            + math.exp(-dx2_b * 100.)
            + math.exp(-dx2_c * 100.)
            )
 
    integ = vegas.Integrator(
        dim * [[0., 1.]], 
        nitn=10, 
        neval=1000,
        analyzer=vegas.reporter(),
        )
    ans = integ(f)

Running this gives the following plausible output::

	(8114594512784433755, 7385927139888736276, 330532203319418968)
	1st estimate of integral = 0.214(37)   chi2/dof = 5.48  Q = 0.0
	2nd estimate of integral = 0.669(19)   chi2/dof = 0.58  Q = 0.8

	(7347371220009087992, 8521783773969794912, 3888315240590098246)
	1st estimate of integral = 0.3094(89)   chi2/dof = 2.85  Q = 0.0
	2nd estimate of integral = 0.3362(22)   chi2/dof = 1.38  Q = 0.2

	(2599668822815729321, 3006539213038182690, 4437560779814560636)
	1st estimate of integral = 0.3336(96)   chi2/dof = 0.85  Q = 0.6
	2nd estimate of integral = 0.3311(23)   chi2/dof = 0.52  Q = 0.9

	(6999262396578332412, 7973831536332976041, 1671032485900695722)
	1st estimate of integral = 0.3147(69)   chi2/dof = 15.25  Q = 0.0
	2nd estimate of integral = 0.3325(22)   chi2/dof = 2.20  Q = 0.0

	(6912587160260435101, 3696713562607419621, 6184517149214074329)
	1st estimate of integral = 0.3277(71)   chi2/dof = 2.29  Q = 0.0
	2nd estimate of integral = 0.3343(22)   chi2/dof = 0.59  Q = 0.8

The first estimate looks unreliable but the second estimate seems 
plausible. As a cross check we run the script again. The integrator
uses different random numbers and gets a completely different result:

	(8114594512784433755, 7385927139888736276, 330532203319418968)
	1st estimate of integral = 0.214(37)   chi2/dof = 5.48  Q = 0.0
	2nd estimate of integral = 0.669(19)   chi2/dof = 0.58  Q = 0.8

Following the advice of the previous section we discard the first
estimate. The second estimate looks fine, and is indeed a reliable
estimate of the integral of *two* of the three peaks in the 
integrand---unfortunately, the integrator has missed one of the 
peaks completely (the one closest to the origin in this case).



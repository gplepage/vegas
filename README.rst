vegas
-----

This package is for evaluating multidimensional integrals using
an improved version of the adaptive Monte Carlo vegas algorithm.
The central package is ``vegas``. A tutorial on its use
is in ``doc/html/index.html`` (or see the online documentation
at ``vegas.readthedocs.io``).

The vegas algorithm has been widely used for decades to evaluate
integrals of 2 or more dimensions numerically. It is particularly
well suited to higher dimensions (*e.g.*, 8 or 13 for Feynman diagram
evaluation). The algorithm in this package is significantly
improved over the original vegas implementation. In particular
a second adaptive strategy has been added. It also supports
multi-processor evaluation of integrands using MPI.

See the ``INSTALLATION`` file for installation directions.
Test ``vegas`` using ``make tests``. Some simple examples are
in the ``examples/`` subdirectory.

Versioning: Version numbers for ``vegas`` are now (2.2 and later) based upon
*semantic  versioning* (http://semver.org). Incompatible changes will be
signaled by incrementing the major version number, where version numbers have
the form major.minor.patch. The minor number signals new features, and the
patch number bug fixes.

| Created by G. Peter Lepage (Cornell University) 2013
| Copyright (c) 2013-2017 G. Peter Lepage

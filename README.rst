vegas
-----

The ``vegas`` package is for evaluating multidimensional integrals using
an improved version of the adaptive Monte Carlo vegas algorithm
(G. P. Lepage, J. Comput. Phys. 27(1978) 192).
A tutorial on its use can be found in the documentation:
see ``doc/html/index.html`` or <https://vegas.readthedocs.io>.

The ``vegas`` algorithm has been widely used for decades to evaluate
integrals of 2 or more dimensions numerically. It is particularly
well suited to higher dimensions (e.g., 9 for Feynman diagram
evaluation). The algorithm in this package is significantly
improved over the original ``vegas`` implementation. In particular
a second adaptive strategy has been added. It also supports
multi-processor evaluation of integrands using MPI.

The new algorithm is described in G. P. Lepage, 
<https://arxiv.org/abs/2009.05112>.

See the ``INSTALLATION`` file for installation directions.
Test ``vegas`` using ``make tests``. Some simple examples are
in the ``examples/`` subdirectory.

``vegas`` version numbers have the form ``major.minor.patch`` where
incompatible changes are signaled by incrementing the ``major`` version
number, the ``minor`` number signals new features, and the ``patch``
number signals bug fixes.


| Created by G. Peter Lepage (Cornell University) 2013
| Copyright (c) 2013-2018 G. Peter Lepage

.. image:: https://zenodo.org/badge/15354897.svg
   :target: https://zenodo.org/badge/latestdoi/15354897

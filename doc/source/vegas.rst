The :mod:`vegas` Package
==================================================================

.. |Integrator| replace:: :class:`vegas.Integrator`
.. |AdaptiveMap| replace:: :class:`vegas.AdaptiveMap`
.. |vegas| replace:: :mod:`vegas`
.. |WAvg| replace:: :class:`lsqfit.WAvg`
.. |chi2| replace:: :math:`\chi^2`
.. |x| replace:: x 
.. |y| replace:: y 

.. moduleauthor:: G. Peter Lepage <g.p.lepage@cornell.edu>

.. automodule:: vegas
    :synopsis: Adaptive multidimensional Monte Carlo integration

Integrator Objects
----------------------------
The central component of the |vegas| package is the integrator class:

.. autoclass:: vegas.Integrator
    
    |Integrator| objects have attributes for each of these parameters.
    In addition they have the following methods:

    .. automethod:: set(ka={}, **kargs)

    .. automethod:: settings(ngrid=0)


AdaptiveMap Objects
---------------------
|vegas|'s remapping of the integration variables is handled
by :class:`AdaptiveMap`.

.. autoclass:: vegas.AdaptiveMap

   .. autoattribute:: dim

   .. autoattribute:: ninc 

   .. attribute:: grid

      The nodes of the grid defining the maps are ``self.grid[d, i]``
      where ``d=0...`` specifies the direction and ``i=0...self.ninc``
      the node.

   .. attribute:: inc

      The increment widths of the grid::

          self.inc[d, i] = self.grid[d, i + 1] - self.grid[d, i]

   .. automethod:: adapt(alpha=0.0, ninc=None)

   .. automethod:: add_training_data(y, f, ny=-1)

   .. automethod:: __call__(y)

   .. automethod:: jac(y)

   .. automethod:: make_uniform(ninc=None)

   .. automethod:: map(y, x, jac, ny=-1)

   .. automethod:: plot_grid(ngrid=40, shrink=False)

   .. automethod:: settings(ngrid=5)

Other Objects
----------------

.. autoclass:: vegas.RunningWAvg

   .. attribute:: mean

      The mean value of the weighted average.

   .. attribute:: sdev

      The standard deviation of the weighted average.
    
   .. autoattribute:: chi2

   .. autoattribute:: dof

   .. autoattribute:: Q

   .. attribute:: itn_results

   A list of the results from each iteration.

   .. automethod:: add(g)

   .. automethod:: summary()

.. autoclass:: vegas.VecIntegrand



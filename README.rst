.. image:: https://travis-ci.org/marekpetrik/RAAM.svg
    :target: https://travis-ci.org/marekpetrik/RAAM

======================================================
RAAM: Robust and Approximate Markov Decision Processes
======================================================

.. role:: python(code)
    :language: python

A simple and easy to use **Python** library to solve Markov decision processes and *robust* Markov decision processes. The library includes mostly helper functions for the CRAAM python interface. In particular, it contains basic simulation routines, approximate dynamic programming through *state aggregation*, and the construction of MDPs from simulation. 

The library supports standard finite or infinite horizon discounted MDPs [Puterman2005]_. The library assumes *maximization* over actions. The states and actions must be finite. 

The robust model extends the regular MDPs [Iyengar2005]_. The library allows to model uncertainty in *both* the transition and rewards, unlike some published papers on this topic. This is modeled by adding an outcome to each action. The outcome is assumed to be minimized by nature, similar to [Filar1997]_.

In summary, the robust MDP problem being solved is:

.. math::

    v(s) = \max_{a \in \mathcal{A}} \min_{o \in \mathcal{O}} \sum_{s\in\mathcal{S}} ( r(s,a,o,s') + \gamma P(s,a,o,s') v(s') ) ~.

Here, :math:`\mathcal{S}` are the states, :math:`\mathcal{A}` are the actions, :math:`\mathcal{O}` are the outcomes. 

The included algorithms are *value iteration* and *modified policy iteration*. The library support both the plain worst-case outcome method and a worst case with respect to a base distribution. The precise algorithms are implemented in C++ in `CRAAM <https://bitbucket.org/marekpetrik/craam>`_; see the project website for a detailed description.

The algorithms that approximate MDPs though robust state aggregation are described in [Petrik2014]_. The robust algorithm generalizes standard state aggregation by capturing the introduced model error through robust models.

Installation
------------

Requirements:

- craam 1.0+ C++ implementation of methods for solving MDPs
- Python 3.5+ (Python 2 is NOT supported)
- Setuptools 7.0
- Numpy 1.8+
- Scipy 0.13 
- Cython 0.21+ 

The package has been tested only on Linux.

Optional dependencies:

- Matplotlib 1.0+ for plotting support

To install, run (use ``--user`` to install locally):

.. code:: bash

    $ python setup.py install

To install in a development mode, execute:

.. code:: bash

    $ python setup.py develop

The development installation will not copy project files to ``site-packages``---any changes to the Python code will be reflected without the need to reinstall.

To test the installation, run the following python code:
    
.. code:: python

    import raam
    import raam.test
    
    raam.test.test()
    
It is also possible to run the tests from the command line as:
    
.. code:: bash

    python -mraam.test

Structure
---------

The project consists of the following main modules:

* :python:`raam.robust` - a pure python implementation of selected robust optimization methods
* :python:`raam.simulator` - framework code for implementing a simulation-based MDP formulation and optimization
* :python:`raam.samples` - methods for handling samples
* :python:`raam.features` - methods for defining state features
* :python:`raam.plotting` - basic plotting support
* :python:`raam.examples` - example MDP domains
* :python:`raam.test` - code unit tests


Methods for solving robust MDPs are provided by :python:`craam.robust`. 

================================  ====================================
Method                            Algorithm
================================  ====================================
:python:`crobust.MDP.vi_gs`      Gauss-Seidel value iteration; runs in a single thread. Computes the worst-case outcome for each action.
:python:`crobust.MDP.vi_jac`     Jacobi value iteration; parallelized with OpenMP. Computes the worst-case outcome for each action.
:python:`crobust.MDP.vi_gs_l1`   The same as ``vi_gs`` except the worst case is bounded with respect to an :math:`L_1` norm.
:python:`crobust.MDP.vi_jac_l1`  The same as ``vi_jac`` except the worst case is bounded with respect to an :math:`L_1` norm.
:python:`crobust.MDP.mpi_jac`    Jacobi modified policy iteration; parallelized with OpenMP. Computes the worst-case outcome for each action. Generally, modified policy iteration is vastly more efficient than value iteration.
================================  ====================================

First Steps
-----------

Solving a Simple MDP
~~~~~~~~~~~~~~~~~~~~

See library CRAAM for a simple example. 

Solving a Sample-based MDP (reinforcement learning)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, define a simulator for a simple MDP chain and sample from it.

.. code:: python

    import raam
    import random
    import itertools

    class StatefulCounter(raam.simulator.Simulator):  
        """
        State: position in chain
        Initial (decision) state: 0
        Actions: {plus, minus}
        Rewards: 90%: next position, 10% this position in chain
        """

        def __init__(self):
            self.state = 0

        @property
        def discount(self):
            return 0.9
            
        def transition(self,state,action):
            decstate = self.state
            
            if action == 'plus': act = 1
            elif action == 'minus': act = -1
            else: raise ValueError('Invalid action')
            
            if random.random() <= 0.9: self.state = state + act
            else: self.state = state            
            return state,self.state 
            
            return self.state
                
        def end_condition(self,state):
            return False
            
        def initstates(self):
            return itertools.repeat(0)
            
        def actions(self,state):
            return ['plus','minus']
            
The next step is to generate samples as follows:

.. code:: python

    horizon = 100
    runs = 5
    sim = StatefulCounter()
    samples = sim.simulate(horizon,sim.random_policy(),runs)
    print("Samples:\n", list(samples.samples()))
    
These samples use the raw state and action representation. The state is in integer in this case, but it could be in principle any python object. So to formulate an MDP, we need to assign unique indices to the states as follows:

.. code:: python

    from craam import crobust
    from raam import features
    
    # transform the samples to discrete samples (build a discrete view)
    # each state (-infinity, +infinity) is assigned a uniques index by IdCache
    sv = raam.SampleView(samples=samples,statemap=features.IdCache(), actmap = features.IdCache(), readonly=True)
    ds = crobust.DiscreteSamples()
    for i in sv.initialsamples():
        ds.add_initial(i)
    for s in sv.samples():
        ds.add_sample(s.statefrom, s.action, s.stateto, s.reward, s.weight, s.step, s.run)

And finally, the following code will actually solve the MDP.

.. code:: python

    # construct MDP from the samples
    smdp = crobust.SampledMDP()
    smdp.add_samples(ds)
    mdp = smdp.get_mdp(0.9)
    print(mdp.mpi_jac())

Note that it is important to map the value function and policy in the last two lines. This is because the sampled robust MDP uses an internal representation that separates decision and expectation states in order to improve computational efficiency.

More examples are provided in the subdirectory ``examples``.

References
----------

.. [Filar1997] Filar, J., & Vrieze, K. (1997). Competitive Markov decision processes. Springer.

.. [Puterman2005] Puterman, M. L. (2005). Markov decision processes: Discrete stochastic dynamic programming. Handbooks in operations research and management …. John Wiley & Sons, Inc.

.. [Iyengar2005] Iyengar, G. N. G. (2005). Robust dynamic programming. Mathematics of Operations Research, 30(2), 1–29. 

.. [Petrik2014] Petrik, M., & Subramanian, D. (2014). RAAM : The benefits of robustness in approximating aggregated MDPs in reinforcement learning. In Neural Information Processing Systems (NIPS).


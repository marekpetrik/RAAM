======================================================
RAAM: Robust and Approximate Markov Decision Processes
======================================================

.. role:: python(code)
    :language: python

A simple and easy to use **Python** library to solve Markov decision processes and *robust* Markov decision processes. The library also contains basic simulation routines, approximate dynamic programming through *state aggregation*, and the construction of MDPs from simulation. 

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

- Python 3.3+ (Python 2 is NOT supported)
- Setuptools 7.0
- Numpy 1.8+
- Scipy 0.13 
- Cython 0.21+ and GCC 4.9+ or LLVM or another compiler with C++11 support

The package has been tested only on Linux, but should also work on Windows and Mac.

Optional dependencies:

- Matplotlib 1.0+ for plotting support

The source code (including the C++ sub-repository) can be downloaded as:

.. code:: bash

    git clone https://marekpetrik@bitbucket.org/marekpetrik/raam.git --recursive

To install, simply execute:

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

* :python:`raam.crobust` - the main algorithms and the python interface to `CRAAM`_
* :python:`raam.robust` - a pure python implementation of selected robust optimization methods
* :python:`raam.simulator` - framework code for implementing a simulation-based MDP formulation and optimization
* :python:`raam.samples` - methods for handling samples
* :python:`raam.features` - methods for defining state features
* :python:`raam.plotting` - basic plotting support
* :python:`raam.examples` - example MDP domains
* :python:`raam.test` - code unit tests


The package :python:`raam.crobust` implements the following algorithms.

================================  ====================================
Method                            Algorithm
================================  ====================================
:python:`crobust.RoMDP.vi_gs`      Gauss-Seidel value iteration; runs in a single thread. Computes the worst-case outcome for each action.
:python:`crobust.RoMDP.vi_jac`     Jacobi value iteration; parallelized with OpenMP. Computes the worst-case outcome for each action.
:python:`crobust.RoMDP.vi_gs_l1`   The same as ``vi_gs`` except the worst case is bounded with respect to an :math:`L_1` norm.
:python:`crobust.RoMDP.vi_jac_l1`  The same as ``vi_jac`` except the worst case is bounded with respect to an :math:`L_1` norm.
:python:`crobust.RoMDP.mpi_jac`    Jacobi modified policy iteration; parallelized with OpenMP. Computes the worst-case outcome for each action. Generally, modified policy iteration is vastly more efficient than value iteration.
================================  ====================================

First Steps
-----------

Solving a Simple MDP
~~~~~~~~~~~~~~~~~~~~

The following code solves a simple (non-robust) MDP problem precisely using modified policy iteration.

.. code:: python

    from raam import crobust

    states = 100
    P1 = np.random.rand(states,states)
    P1 = np.diag(1/np.sum(P1,1)).dot(P1)
    P2 = np.random.rand(states,states)
    P2 = np.diag(1/np.sum(P2,1)).dot(P2)
    r1 = np.random.rand(states)
    r2 = np.random.rand(states)
    
    transitions = np.dstack((P1,P2))
    rewards = np.column_stack((r1,r2))
    actions = np.array((0,1))
    outcomes = np.array((0,0))
    
    rmdp = crobust.RoMDP(states,0.99)
    rmdp.from_matrices(transitions,rewards,actions,outcomes)
    value,policy,residual,iterations = rmdp.mpi_jac(100)

    print('Value function', value)

This example could be easily converted to a robust MDP by appropriately defining additional outcomes (the options available to nature) with transition matrices and rewards.

Solving a Sample-based MDP (reinforcement learning)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, define a simulator for a simple counter MDP. There are two types of states in the simulated MDP: decision and expectation states. The decision state is the standard MDP state, while the expectation state represents a post-decision state. The evolution of the process alternates between decision and expectation states.

.. code:: python
    
    import raam
    import random

    class StatefulCounter(raam.simulator.StatefulSimulator):  
        """
        Decision state: position in chain
        Expectation state: position in chain, change (+1,-1)
        Initial (decision) state: 0
        Actions: {plus, minus}
        Rewards: 90%: next position, 10% this position in chain
        """

        def __init__(self):
            self.state = 0

        @property
        def discount(self):
            return 0.9
            
        def transition_dec(self,action):
            decstate = self.state
            
            if action == 'plus':
                self.state = (decstate, +1)
            elif action == 'minus':
                self.state = (decstate, -1)
            else:
                raise ValueError('Invalid action')
            
            return self.state
            
        def transition_exp(self):        
            pos,act = self.state

            if random.random() <= 0.9:
                self.state = pos + act
            else:
                self.state = pos            
            return pos,self.state            
                
        def end_condition(self,decstate):
            return False
            
        def reinitstate(self,param):
            self.state = 0
            return self.state
            
        def actions(self):
            return ['plus','minus']

This is an example of a stateful simulator class based on :python:`raam.simulator.StatefulSimulator`. Stateless simulator that allow to model transitions starting in arbitrary states can be based on :python:`raam.simulator.Simulator`.


The next step is to generate samples as follows:

.. code:: python

    horizon = 100
    runs = 5
    sim = StatefulCounter()
    samples = sim.simulate(horizon,sim.random_policy(),runs)
    
And finally, the samples are used to construct an sampled robust MDP. Even and odd states are aggregated together. :python:`craam.SRoMDP` is a sampled version of the robust MDP.

.. code:: python

    from raam import crobust
    from raam import features
    r = crobust.SRoMDP(2,0.9)
    
    aggregation = lambda s: s // 2  # aggregate states
    idnt = lambda s: s              # assume the worst-case behavior of individual states
    expcache = features.IdCache()    # treat every expectation state separately
    actcache = features.IdCache()    # treat every action separately
    r.from_samples(samples,decagg_big=aggregation,decagg_small=idnt,
                    expagg=expcache,actagg=actcache)
    
    r.rmdp.set_uniform_distributions(1.0)   # define uniform distributions for norm bounds
    val,pol = r.rmdp.mpi_jac_l1(100)[:2]
    # map value function 
    val = r.decvalue(12,val,minstate=-6)
    pol = r.decpolicy(12,pol,minstate=-6)


Note that it is important to map the value function and policy in the last two lines. This is because the sampled robust MDP uses an internal representation that separates decision and expectation states in order to improve computational efficiency.

More examples are provided in the subdirectory ``examples``.

References
----------

.. [Filar1997] Filar, J., & Vrieze, K. (1997). Competitive Markov decision processes. Springer.

.. [Puterman2005] Puterman, M. L. (2005). Markov decision processes: Discrete stochastic dynamic programming. Handbooks in operations research and management …. John Wiley & Sons, Inc.

.. [Iyengar2005] Iyengar, G. N. G. (2005). Robust dynamic programming. Mathematics of Operations Research, 30(2), 1–29. 

.. [Petrik2014] Petrik, M., & Subramanian, D. (2014). RAAM : The benefits of robustness in approximating aggregated MDPs in reinforcement learning. In Neural Information Processing Systems (NIPS).


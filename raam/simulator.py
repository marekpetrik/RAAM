""" 
===========================================
Simulator tools (:mod:`raam.simulator`)
===========================================

This module also provides a simple framework for implementing a simulator that 
can be used to generate samples. In particular, a simulator must implement an 
interface :class:`raam.simulator.Simulator`. A template for the simulator is::
    class SimulatorTemplate(Simulator):  
        @property
        def discount(self):
            pass
        def transition(self,state,action):
            pass
        def end_condition(self,state):
            pass
        def initstates(self):
            pass
        def actions(self,state):
            pass
            
:class:`raam.simulator.Simulator` is generally a state-less simulator. It does not
need to track the state of the simulation.    
        
:class:`raam.simulator.SimpleSimulator` is generally a stateful simulator. It 
needs to track the current state of the simulation.
"""
import numpy as np
import abc
import raam.samples
from raam.samples import Sample
import itertools
import collections
import random
import numbers

class Simulator(metaclass=abc.ABCMeta):
    """
    Represents a simulator class that can be used to generate samples.
    The class must be inherited from to define the actual simulator.

    The simulation is based on two types of states: 
        1. Decision states - states in which decisions are taken and the  deterministic 
            transition is to expectation states
        2. Expectation states - states from which the transition is stochastic 
            into decision states. The reward is also determined in this transition.

    See also :class:`StatefulSimulator` for a simulator framework that does not support
    sampling arbitrary states.

    The representation of the decision and expectation states may be arbitrary objects.

    The simulator can also represent a stateful process. Reading the initial
    state would reset the simulation to the beginning.
    """
    
    @abc.abstractmethod
    def transition(self,state,action):
        """
        Computes the transition from a decision state to an expectation state. 
        This is a deterministic function and it must return the same value 
        every time it is called.

        Parameters
        ----------
        state : object
            Domain dependent representation of a decision state
        action : object
            Domain dependent representation of an action

        Returns
        -------
        reward : float
            Reward associated with the transition
        state : object
            Domain dependent representation of a decision state
        """
        pass

    @abc.abstractproperty
    @property
    def discount(self):
        """ Returns the discount factor. """
        pass

    @abc.abstractmethod
    def actions(self,state):
        """ Returns the list or an iterator of applicable actions (when finite). 
            It returns ``None`` if there are too many to enumerate. """
        pass

    @abc.abstractmethod
    def initstates(self):
        """ 
        Returns a list or a generator (possibly unbounded) of the initial 
        decision states.

        Reading another initial state may reset the current state
        of the simulator (if it is stateful).
        """
        pass
        
    def end_condition(self,state):
        """
        Returns true if the simulation should stop after reaching state

        Parameters
        ----------
        state : state
            Decision state to be checked

        Returns
        -------
        out : bool
            Returns true if the simulation should stop.

        Notes
        -----
        The default implementation always returns False
        """
        return False

    def compute_value(self, state, action, valuefun, samplecount=10):
        """ 
        Computes the value of an expectation state

        Parameters
        ----------
        state :  State
            Starting state
        action : Action
            Action taken
        valuefun :  function(state -> float) 
            Value function for decision states
        samplecount : int 
            The number of samples to be used in computing the expected 
            value function

        Returns
        -------
        out : float
            Value of the expectation state
        """
        result = 0
        for s in range(samplecount):
            profit, nxtstate = self.transition(state,action)
            result += profit
            result += self.discount * valuefun(nxtstate)
        return result/samplecount

    def simulate(simulator,horizon,policy,runs,initstates=None,
                end_condition=None,startsteps=0,samples=None,
                probterm=None,transitionlimit=None):
        """ 
        Run simulation using the provided policy 
    
        Parameters
        ----------
        horizon : int
            Limit on the number of steps simulated
        policy : function (state -> action)
            Function that returns the action number for each state
        runs : int
            The number of the run if an integer, or list of multiple 
            numbers to get multiple runs
        initstates : iterator, (optional)
            Iterator that is used to generate a new state for every run. 
            If not provided then the default initial states of the 
            simulator are used. Use itertools.repeat to use a single state.
        end_condition : function (state->bool) (optional)
            A function that takes a decision state and returns true if 
            a terminal state is reached.
        startsteps : int, list of ints (optional)
            The simulation will start in this step (for each run)
        samples : raam.samples.Samples, optional
            Sample storage. If None, then a memory backed sample 
            storage is created and used.
        probterm : float, optional
            Simulation termination probability in each step. Can be set to 
            1-``discount`` to simulate the behavior with discount 
            representing the termination probability.
        transitionlimit : int, optional
            Stops simulation when this number of transitions sampled is 
            reached.  No limit when omitted.
    
        Returns
        -------
        out : raam.Samples
            A representation of the samples
            
        Remarks
        -------
        Even when ``initstates`` are provided, it is necessary to set 
        the number of runs appropriately. The number of total simulation 
        runs is limited by the minimum of ``runs`` and the length 
        of ``initstates``.
        """
        # from state to state-action
        if samples is None:
            samples = raam.samples.MemSamples()
    
        if isinstance(runs,numbers.Integral):
            runs = range(runs)
        elif not isinstance(runs,collections.Iterable):
            raise ValueError("Parameter 'runs' must be either int or iterable")
            
        if initstates is None:
            initstates = simulator.initstates()
    
        if isinstance(startsteps,numbers.Integral):
            startsteps = itertools.repeat(startsteps, len(runs))
    
        if end_condition is None:
            end_condition = simulator.end_condition
    
        if not isinstance(initstates,collections.Iterable):
            raise ValueError("Parameter 'initstates' must be iterable")
    
        # the total number of transitions sampled
        transitions = 0
    
        for di,run,step in zip(initstates,runs,startsteps):
            
            decstate = di
            samples.add_initial(decstate)
    
            for i in range(horizon):
                if end_condition(decstate):
                    break
                if transitionlimit is not None and transitions >= transitionlimit:
                    break

                action = policy(decstate)
                profit,newstate = simulator.transition(decstate, action)
                samples.add_sample(Sample(decstate, action, newstate, profit, 1.0, i+step, run))
                                    
                decstate = newstate
                # test the termination probability only after at least one transition
                if probterm is not None and random.random() <= probterm:
                    break
                
                transitions += 1
            
            if transitionlimit is not None and transitions >= transitionlimit:
                break

        return samples
    
    def greedy_policy_v(simulator,value,samplecount,features=None,actionset=None):
        """
        Computes a greedy policy for the provided **decision state** value function.
    
        Parameters
        ----------
        value : {function(state -> float), array_like}
            Decision state value function used to compute the policy.
            If a list is provided then it is assumed to be coefficients
            and the appropriate function is constructed using the simulator.
        samplecount : int
            Number of samples to use in estimating the value of an expectation 
            state from samples of transitions to decision states.
        actionset : set, optional
            Set of actions considered when computing the policy.
            When not provided or None, the default actions from the
            simulator are used
        features : function (state -> features)
            Function that computes the features for the state
    
        Returns
        -------
        out : function(state -> action)
            Policy that is greedy with respect to the provided value function
    
        Notes
        -----
        This function requires a known stochastic transition model and may be quite 
        slow when there are many actions and samples (samplecount) is large. 
        """
    
        if features is None:
            features = lambda x: x
    
        if actionset is None:
            actionset = lambda s: simulator.actions(s)
        else:
            actionset = lambda s: actionset
    
        if not hasattr(value, '__call__'):
            cx = value
            value = lambda s : np.dot( features(s), cx )
    
        def policy(decstate):
            maxvalue = -float('inf')
            maxaction = None
            for a in actionset(decstate):
                v = simulator.compute_value(decstate, a, value, samplecount)
                if v > maxvalue:
                    maxaction = a
                    maxvalue = v
            return maxaction
        return policy
    
    def random_policy(simulator,actionset=None):
        """
        Returns a random policy for a problem with a finite (and small)
        number of actions.

        Parameters
        ----------
        actionset : set (optional)
            Set of actions considered when computing the policy.
            When not provided or None, the default actions from the
            simulator are used
    
        Returns
        -------
        out : function(state -> action)
            A uniformly random policy that is greedy with respect to the provided 
            value function.
    
        """
        if actionset is None:
            def policy(s): 
                aset = simulator.actions(s)
                return aset[ np.random.randint( 0,len(aset) ) ]
            return policy
        else:
            return lambda s: actionset[np.random.randint( 0,len(actionset) ) ]


    def sample_transitions(simulator,samples,count,append=True):
        """
        Generates additional transition samples for each state and action that
        is present in the data. 
    
        Parameters
        ----------
        samples : dict 
            Collection of samples being expanded. This object is not modified.
        count : int
            The number of additional samples to generate for each 
            expectation state.
        append : bool, optional
            Whether to append the original expectation samples or if they 
            should be discarded. Default is true.
    
        Returns
        -------
        out : samples.Samples 
            Original samples with the additional expectation states augmented
    
        Notes
        -----
        The function does not check for duplicates. It is possible there 
        are multiple samples then.
        """
    
        new_samples = samples.copy(samples=append,initial=True)
        
        for e in samples.samples():
            expstate = e.statefrom
            action = e.action
            
            for sample in range(count):
                profit,decstate = simulator.transition(expstate,action)
                new_samples.add_sample(Sample(expstate,action,decstate,\
                                            profit, 1.0, e.step, e.run))
        return new_samples
    
    def sample_actions_transitions(simulator,samples,policy=None,count=1,
                    append=True):
        """
        Generates additional actions for states
        in the samples and also samples appropriate transitions.
    
        Parameters
        ----------
        samples : dict 
            Collection of samples being expanded. This object is not modified.
        policy : function (state -> actions)
            The policy used to generate the new samples. If None, then all
            actions are used.
            Policy is assumed to return multiple actions, ot a tuple of
            length 1.
        count : int
            The number of additional samples to generate for each 
            expectation state.
        append : bool, optional
            Whether to append the original expectation samples or if they 
            should be discarded. Default is true.
    
        Returns
        -------
        out : samples.Samples 
            Original samples with the additional decision states augmented
    
        Notes
        -----
        The function does not check for duplicates. It is very likely there 
        are multiple samples then with the same state and action.
        """
        new_samples = samples.copy(samples=append,initial=True)
    
        # transform the policy to generate a list of actions
        # all actions when None
        if policy is None:
            policylist = lambda ds : simulator.actions(ds)
        else:
            policylist = lambda ds : [policy(ds)]
    
        for d in samples.samples():
            decstate = d.statefrom
            actions = policylist(decstate)
            for action in actions:
                for sample in range(count):
                    reward,newstate = simulator.transition(decstate,action)
                    new_samples.add_sample(Sample(decstate, action, newstate,\
                            reward, 1.0, d.step, d.run))
        return new_samples
    
def vec2policy(policy_vec,actions,decagg,noaction=None):
    """
    Computes a policy function from a vector of probabilities for actions for
    each aggregate state.
    
    Parameters
    ----------
    policy_vec : numpy.ndarray (#states x #actions), 
                or (#states) for a deterministic policy.
        Probability of actions for each decision state, or the index of the
        action. If it is a deterministic policy and the index is -1,
        the policy is assumed to be undefined.
    actions : list
        List of actions for the policy. This list must match the list from the 
        samples and *NOT* from the simulator. The order and the number of 
        the simulator actions may be different from what is observed in the 
        sample.
    decagg : function (state -> int)
        A function that assigns the number of the aggregate state and its index
        within the aggregation to each individual sample *decision* state.
    noaction : action, optional
        Action index to take in states that do not have a defined policy. The policy 
        is considered to be undefined when it is negative. This only works for 
        deterministic policies.
    
    Returns
    -------
    policy : function (state -> action)
        A policy that respects the given aggregation and policy vector
        
    Remarks
    -------
    If raam.crobust.SRoMDP is used to construct a robust MDP and a policy, then
    it is necessary to transform the outputs of raam.crobust.RoMDP using 
    raam.crobust.SRoMDP.decpolicy first.
    """
    
    if len(policy_vec.shape) == 2:
        # we have a *randomized* policy here
        if len(actions) != policy_vec.shape[1]:
            raise ValueError('Policy vector and action dimensions do not match.')
        
        if np.min(np.max(policy_vec,1)) < 0.999:
            # the policy is randomized
            policy = lambda decstate: np.random.choice(actions,1,\
                                p=policy_vec[decagg(decstate)])[0]
        else:
            # the policy is deterministic
            bestactions = np.argmax(policy_vec,1)
            policy = lambda decstate: actions[bestactions[decagg(decstate)]]
    
    elif len(policy_vec.shape) == 1:
        # we have a *deterministic* policy here
        if np.max(policy_vec) >= len(actions):
            raise ValueError("Action index %d prescribed by the policy does not exist." % np.max(policy_vec))
        if np.min(policy_vec) < -1:
            raise ValueError("Negative action indices less than -1 (undefined) are not allowed.")

        if noaction is None:
            def policy(decstate):
                actionid = policy_vec[decagg(decstate)]
                if actionid < 0:
                    raise RuntimeError('Undefined action (-1) for state %s.' % str(decstate))
                return actions[actionid]
            return policy
        else:            
            def policy(decstate):
                actionid = policy_vec[decagg(decstate)]
                if actionid < 0:
                    actionid = noaction
                return actions[actionid]
            return policy
    else:
        raise ValueError('The policy vector must be 1 or 2 dimensional.')
        
    return policy
 
def construct_mdp(sim, discount):
    """
    Uses the simulator `sim` to construct an MDP model. 
    
    Simulator can provide any hashable states. States in the MDP model are
    identified by an integral index based on the order returned 
    by `sim.all_states`. Actions are also indexed sequentially based on the
    order returned by `sim.actions`.

    The simulator `sim` must support a function: 
        - `all_transitions`, which takes a state and action and generates a sequence of:
            `(nextstate,probability,reward), (nextstate, probability, reward), ....`
        - `all_states`, which returns a sequence of all possible states in the
                        problem and the initial probability in for that state:
                        `(s,p0), (s,p0), ...`
        - `initial_distribution`, which returns the initial distribution for all 
                                all states returned be `all_states`

    Parameters
    ----------
    sim : raam.Simulator
        Needs functions: all_transitions and all_states 
    discount : float 
        Discount factor
        
    Returns
    -------
    mdp : craam.MDP
        Resulting MDP
    p0 : array like
        Initial distribution
    all_states : list
        List of all states
    """

    allstates = sim.all_states()
    initprobs = sim.initial_distribution()

    if abs(np.sum(initprobs) - 1) > 1e-5:
        raise ValueError('Initial distribution does not sum to one')
    
    state2id = {s:i for i,s in enumerate(allstates)}
    
    mdp = craam.MDP(len(allstates),discount)
    
    # associate states with ids
    for fstateid,fstate in enumerate(allstates):
        
        if sim.end_condition(fstate):
            # do not add any transition or actions for the terminal state
            # such state is automatically then treated as terminal
            continue

        for actionid, action in enumerate(sim.actions(fstate)):
            for nstate, prob, rew in sim.all_transitions(fstate, action):
                nstateid = state2id[nstate]
                mdp.add_transition(fstateid,actionid,nstateid,prob,rew)            
    
    p0 = np.array(initprobs)
    
    return mdp, p0

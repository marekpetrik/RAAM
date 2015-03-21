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
        def transition_dec(self,decstate,action):
            pass
        def transition_exp(self,expstate):
            pass
        def end_condition(self,decstate):
            pass
        def initstates(self):
            pass
        def actions(self,decstate):
            pass
            
:class:`raam.simulator.Simulator` is generally a state-less simulator. It does not
need to track the state of the simulation.    
            
An alternative is to implement the simple simulator interface 
:class:`raam.simulator.StatefulSimulator`::
    class StatefulSimulatorTemplate(SimpleSimulator):  
        @property
        def discount(self):
            pass
        def transition_dec(self,action):
            pass
        def transition_exp(self):
            pass
        def end_condition(self,decstate):
            pass
        def reinitstate(self,parameters):
            pass
        def actions(self):
            pass
        
:class:`raam.simulator.SimpleSimulator` is generally a stateful simulator. It 
needs to track the current state of the simulation.
"""
import numpy as np
import abc
import raam.samples
from raam.samples import DecSample, ExpSample
import itertools
import collections
import random

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

    See Also
    --------
    StatefulSimulator
    """
    
    @abc.abstractmethod
    def transition_dec(self,decstate,action):
        """
        Computes the transition from a decision state to an expectation state. 
        This is a deterministic function and it must return the same value 
        every time it is called.

        Parameters
        ----------
        decstate : object
            Domain dependent representation of a decision state
        action : object
            Domain dependent representation of an action

        Returns
        -------
        out : object
            Domain dependent representation of an expectation state.
        """
        pass

    @abc.abstractmethod
    def transition_exp(self,expstate):
        """
        Computes the transition from an expectation state to a decision state. 
        This is a stochastic function and it may return a different value 
        every time it is called.

        Parameters
        ----------
        expstate : object
            Domain-dependent representation of an expectation state

        Returns
        -------
        reward : float
            Reward associated with the transition
        decstate : object
            Domain dependent representation of a decision state
            
        See Also
        --------
        Simulator.transition_exp_p
        """
        pass
   
    @abc.abstractproperty
    @property
    def discount(self):
        """ Returns the discount factor. """
        pass

    @abc.abstractmethod
    def actions(self,decstate):
        """ Returns the list or an iterator of applicable actions (when finite). 
            It returns ``None`` if there are too many to enumerate. """
        pass

    @abc.abstractmethod
    def initstates(self):
        """ 
        Returns a list or a generator (possibly unbounded) of the initial 
        decision states.
        """
        pass
        
    def end_condition(self,decstate):
        """
        Returns true if the simulation should stop after reaching decstate

        Parameters
        ----------
        decstate : decstate
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


    def compute_value(self, expstate, value, samplecount=10):
        """ 
        Computes the value of an expectation state

        Parameters
        ----------
        expstate :  expectation state
            The expectation state
        value :  function(decState -> float) 
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
            profit, decstate = self.transition_exp(expstate)
            result += profit
            result += self.discount * value(decstate)
        return result/samplecount

    def simulate(simulator,horizon,policy,runs,initstates=None,end_condition=None,
                 startsteps=0,samples=None,probterm=None):
        """ 
        Run simulation using the provided policy 
    
        Parameters
        ----------
        horizon : int
            Limit on the number of steps simulated
        policy : function (decstate -> action)
            Function that returns the action number for each state
        runs : int
            The number of the run if an integer, or list of multiple numbers 
            to get multiple runs
        initstates : iterator, (optional)
            Iterator that is used to generate a new state for every run. If 
            not provided then the default initial states of the simulator 
            are used. Use itertools.repeat to use a single state.
        end_condition : function (decstate->bool) (optional)
            A function that takes a decision state and returns true if 
            a terminal state is reached.
        startsteps : int, list of ints (optional)
            The simulation will start in this step (for each run)
        samples : raam.samples.Samples, optional
            Sample storage. If None, then a memory backed sample storage is 
            created and used.
        probterm : float, optional
            Simulation termination probability in each step. Can be set to 1-discount to
            simulate the behavior with discount representing the termination
            probability.
    
        Returns
        -------
        out : raam.Samples
            A representation of the samples
            
        Remarks
        -------
        Even when ``initstates`` are provided, it is necessary to set the number of 
        runs appropriately. The number of total simulation runs is limited by the 
        minimum of ``runs`` and the length of ``initstates``.
        """
        # from state to state-action
        if samples is None:
            samples = raam.samples.MemSamples()
    
        if type(runs) == int:
            runs = range(runs)
        
        if initstates is None:
            initstates = simulator.initstates()
    
        if type(startsteps) == int:
            startsteps = itertools.repeat(startsteps, len(runs))
    
        if end_condition is None:
            end_condition = simulator.end_condition
    
        if not isinstance(initstates,collections.Iterable):
            raise ValueError('Initstates must be iterable')
    
        for di,run,step in zip(initstates,runs,startsteps):
            
            decstate = di
            samples.add_initial(decstate)
    
            for i in range(horizon):
                
                if end_condition(decstate):
                    break

                action = policy(decstate)
                expstate = simulator.transition_dec(decstate, action)
                samples.add_dec(DecSample(decstate, action, expstate, i+step, run))
                                    
                profit,decstate = simulator.transition_exp(expstate)
                samples.add_exp(ExpSample(expstate, decstate, profit, 1.0, i+step, run))

                # test the termination probablity only after at least one transition
                if probterm is not None:
                    if random.random() <= probterm:
                        break

        
        return samples
    
    def greedy_policy_v(simulator,value,samplecount,features=None,actionset=None):
        """
        Computes a greedy policy for the provided **decision state** value function.
    
        Parameters
        ----------
        value : {function(decstate -> float), array_like}
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
        features : function (expstate -> features)
            Function that computes the features for the state
    
        Returns
        -------
        out : function(decstate -> action)
            Policy that is greedy with respect to the provided value function
    
        Notes
        -----
        This function requires a known stochastic transition model and may be quite 
        slow when there are many actions and samples (samplecount) is large. 
        
        Hard to use when the optimal action must be computed using LP.
    
        See Also
        --------
        greedy_policy_q
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
                expstate = simulator.transition_dec(decstate, a)
                v = simulator.compute_value(expstate, value, samplecount)
                if v > maxvalue:
                    maxaction = a
                    maxvalue = v
            return maxaction
        return policy
    
    def greedy_policy_q(simulator,qvalue,features=None,actionset=None):
        """
        Computes a greedy policy for the provided **expectation state** value 
        function.
    
        Parameters
        ----------
        qvalue : {function(expstate -> float), array_like}
            Expectation state value function used to compute the policy.
            If a list is provided then it is assumed to be coefficients
            and the appropriate function is constructed using the simulator.
        actionset : set, optional
            Set of actions considered when computing the policy.
            When not provided or None, the default actions from the
            simulator are used
        features : function (expstate -> features), optional
            Function that computes the features for the state
    
        Returns
        -------
        out : function(decstate -> action)
            Policy that is greedy with respect to the provided value function
    
        Notes
        -----
        This function does not require a known stochastic transition model, but
        the transitions to expectation states from decision states must be known.
        
        Easy to use when the optimal action must be computed using LP.
    
        See Also
        --------
        greedy_policy_v
        """
        if features is None:
            features = lambda x : x
    
        if actionset is None:
            actionset = lambda s: simulator.actions(s)
        else:
            actionset = lambda s: actionset
    
        if not hasattr(qvalue, '__call__'):
            cx = qvalue
            qvalue = lambda s : np.dot(features(s),cx)
    
        def policy(decstate):
            maxvalue = -float('inf')
            maxaction = None
            for a in actionset(decstate):
                expstate = simulator.transition_dec(decstate, a)
                v = qvalue(expstate)
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
        out : function(decstate -> action)
            A uniformly random policy that is greedy with respect to the provided 
            value function.
    
        """
        if actionset is None:
            def policy(s): 
                aset = simulator.actions(s)
                return aset[ np.random.randint( 0,len(aset) ) ]
            return policy
        else:
            return lambda s: actionset[ np.random.randint( 0,len(actionset) ) ]


    def sample_exp_ofexp(simulator,samples,count,append=True):
        """
        Generates additional samples for *expectation* states. The new samples
        are generated from the sampled expectation states (source).
    
        Parameters
        ----------
        samples : dict 
            Collection of samples being expanded. This object is not modified.
        count : int
            The number of additional samples to generate for each expectation state.
        append : bool (optional, True)
            Whether to append the original expectation samples or if they 
            should be discarded. The samples are appended to a new object
    
        Returns
        -------
        out : dict 
            Original samples with the additional expectation states augmented
    
        Notes
        -----
        The function does not check for duplicates. It is possible there are multiple
        samples then.
        """
    
        new_samples = samples.copy(dec=True,exp=append,initial=True)
        
        for e in samples.expsamples():
            expstate = e.expStateFrom
    
            for sample in range(count):
                profit,decstate = simulator.transition_exp(expstate)
                new_samples.add_exp(ExpSample(expstate,decstate,\
                                            profit, 1.0, e.step, e.run))
        return new_samples
    
    def sample_exp_ofdec(simulator,samples,count,append=True):
        """
        Generates additional samples for *expectation* states. The new samples
        are generated from the sampled decision states (target).
    
        Parameters
        ----------
        samples : dict 
            Collection of samples being expanded. This object is not modified.
        count : int
            The number of additional samples to generate for each decision state 
        append : bool (optional, True)
            Whether to append the original expectation samples or if they should be discarded.
    
        Returns
        -------
        out : dict 
            Original samples with the additional expectation states augmented
    
        Notes
        -----
        The function does not check for duplicates. It is possible there are multiple
        samples then.
        """
        new_samples = samples.copy(dec=True,exp=append,initial=True)
        
        for d in samples.decsamples():
            expstate = d.expStateTo
    
            for sample in range(count):
                profit,decstate = simulator.transition_exp(expstate)
                new_samples.add_exp(ExpSample(expstate, decstate, profit, 1.0, \
                    d.step, d.run))
        
        return new_samples
    
    def sample_dec_ofdec(simulator,samples,policy=None,append=True):
        """
        Generates additional *decision* states samples from decision samples.
    
        Parameters
        ----------
        samples : dict 
            Collection of samples being expanded. This object is not modified.
        policy : function (decstate -> action)
            The policy used to generate the new samples. If None, then all
            actions are used
        append : bool (optional, True)
            Whether to append the original decision samples or if they should be discarded.
    
        Returns
        -------
        out : dict 
            Original samples with the additional decision states augmented
    
        Notes
        -----
        The function does not check for duplicates. It is very likely there are multiple
        samples then with the same state and action.
        """
        
        new_samples = samples.copy(dec=append,exp=True,initial=True)
    
        # transform the policy to generate a list of actions
        # all actions when None
        if policy is None:
            policylist = lambda ds : simulator.actions(ds)
        else:
            policylist = lambda ds : [policy(ds)]
    
        for d in samples.decsamples():
            decstate = d.decStateFrom
            action_original = d.action
    
            actions = policylist(decstate)
            for action in actions:
                if action == action_original and append: continue
                expstate = simulator.transition_dec(decstate,action)
                new_samples.add_dec(DecSample(decstate, action, expstate,\
                            d.step, d.run))
        
        return new_samples
    
    def sample_dec_ofexp(simulator,samples,policy=None,append=True):
        """
        Generates additional *decision* states samples from decision samples.
    
        Parameters
        ----------
        samples : dict 
            Collection of samples being expanded. This object is not modified.
        policy : function (decstate -> action)
            The policy used to generate the new samples. If None, then all
            actions are used
        append : bool (optional, True)
            Whether to append the original decision samples or if they should be discarded.
    
        Returns
        -------
        out : dict 
            Original samples with the additional decision states augmented
    
        Notes
        -----
        The function does not check for duplicates. It is very likely there are multiple
        samples then with the same state and action.
        """
        
        new_samples = samples.copy(dec=append,exp=True,initial=True)
    
        # transform the policy to generate a list of actions
        # all actions when None
        if policy is None:
            policylist = lambda ds : simulator.actions(ds)
        else:
            policylist = lambda ds : [policy(ds)]
    
        for e in samples.expsamples():
            decstate = e.decStateTo
    
            actions = policylist(decstate)
            for action in actions:
                expstate = simulator.transition_dec(decstate,action)
                new_samples.add_dec(DecSample(decstate, action, expstate, e.step, e.run+1))
        
        return new_samples


class StatefulSimulator(metaclass=abc.ABCMeta):
    """
    Represents a simulator class that can be used to generate samples.
    The class must be inherited from to define the actual simulator.
    
    This simulator only supports restarts and continuation from the previous state. 
    The simulator cannot be stateless. See the class :class:`Simulator` for a 
    simulator with greater control over sampling.

    The representation of the decision and expectation states may be arbitrary,
    but they must be convertible to lists/vectors of real numbers using ``decstate_rep``
    and ``expstate_rep``.

    Notes
    -----
    Expectation states must hashable (immutable, not a list) to use 
    ``sample_exp_ofexp`` and ``sample_exp_ofdec``. 

    See Also
    --------
    Simulator
    """

    @abc.abstractmethod
    def transition_dec(self,action):
        """
        Computes the transition from a decision state to an expectation state. 
        This is a deterministic function and it must return the same value 
        every time it is called.

        Parameters
        ----------
        action : object
            Domain dependent representation of an action

        Returns
        -------
        out : object
            Domain dependent representation of an expectation state.
        """
        pass

    @abc.abstractmethod
    def transition_exp(self):
        """
        Computes the transition from an expectation state to a decision state. 
        This is a stochastic function and it may return a different value 
        every time it is called.

        Returns
        -------
        reward : float
            Reward associated with the transition
        decstate : object
            Domain dependent representation of a decision state
            
        See Also
        --------
        Simulator.transition_exp_p
        """
        pass
    
    @abc.abstractproperty
    @property
    def discount(self):
        """ Returns the discount factor. """
        pass

    @abc.abstractmethod
    def actions(self):
        """ Returns the list or an iterator of applicable actions (when finite). 
            It returns ``None`` if there are too many to enumerate. """
        pass

    @abc.abstractmethod
    def reinitstate(self,parameters = None):
        """ 
        Returns a list or a generator (possibly unbounded) of the initial 
        decision states.
        
        Parameters
        ----------
        parameters: object, optional
            The initialization parameters digested by the simulator.
            
        Returns
        -------
        out : decstate
            Initial decision state
        """
        pass
    
    def end_condition(self,decstate):
        """
        Returns true if the simulation should stop after reaching decstate

        Parameters
        ----------
        decstate : decstate
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

    def simulate(simulator,horizon,policy,runs,initparams=None,end_condition=None,
                    startsteps=0,samples=None,probterm=None):
        """ 
        Run simulation using the provided policy.
    
        Parameters
        ----------
        horizon : int
            Limit on the number of steps simulated
        policy : function (decstate -> action)
            Function that returns the action number for each state
        runs : int
            The number of the run if an integer, or list of multiple numbers 
            to get multiple runs
        initparams : iterator, (optional)
            Iterator that is used to generate a new state for every run. If 
            not provided then the default initial states of the simulator 
            are used. Use itertools.repeat to use a single state.
        end_condition : function (decstate->bool) (optional)
            A function that takes a decision state and returns true if 
            a terminal state is reached.
        startsteps : int, list of ints (optional)
            The simulation will start in this step (for each run)
        samples : raam.samples.Samples, optional
            Sample storage. If None, then a memory backed sample storage is 
            created and used.
        probterm : float, optional
            Simulation termination probability in each step. Can be set to 1-discount to
            simulate the behavior with discount representing the termination
            probability.
    
        Returns
        -------
        out : raam.Samples
            A representation of the samples
            
        Remarks
        -------
        Even when ``initparams`` are provided, it is necessary to set the number of 
        runs appropriately. The number of total simulation runs is limited by the 
        minimum of ``runs`` and the length of ``initparams``.
        """

        if samples is None:
            samples = raam.samples.MemSamples()
    
        if type(runs) == int:
            runs = range(runs)
        
        if initparams is None:
            initparams = itertools.repeat(None)
    
        if type(startsteps) == int:
            startsteps = itertools.repeat(startsteps, len(runs))
    
        if end_condition is None:
            end_condition = simulator.end_condition
    
        if not isinstance(initparams,collections.Iterable):
            raise ValueError('Initparams must be iterable')
    
        for param,run,step in zip(initparams,runs,startsteps):
            
            decstate = simulator.reinitstate(param)
            samples.add_initial(decstate)
    
            for i in range(horizon):    
                if end_condition(decstate):
                    break
                
                action = policy(decstate)
                expstate = simulator.transition_dec(action)
                samples.add_dec(DecSample(decstate, action, expstate, i+step, run))
                
                profit,decstate = simulator.transition_exp()
                samples.add_exp(ExpSample(expstate, decstate, profit, 1.0, i+step, run))

                # test the termination probablity only after at least one transition
                if probterm is not None:
                    if random.random() <= probterm:
                        break
        
        return samples
    
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
        out : function(decstate -> action)
            A uniformly random policy that is greedy with respect to the provided 
            value function.
    
        """
        if actionset is None:
            def policy(s): 
                aset = simulator.actions()
                return aset[ np.random.randint( 0,len(aset) ) ]
            return policy
        else:
            return lambda s: actionset[ np.random.randint( 0,len(actionset) ) ]


def vec2policy(policy_vec,actions,decagg,noaction=None):
    """
    Computes a policy function from a vector of probabilities for actions for
    each aggregate state.
    
    Parameters
    ----------
    policy_vec : numpy.ndarray (#decstates x #actions), 
                or (#decstates) for a deterministic policy.
        Probability of actions for each decision state, or the index of the
        action. If it is a deterministic policy and the index is -1,
        the policy is assumed to be undefined.
    actions : list
        List of actions for the policy. This list must match the list from the 
        samples and *NOT* from the simulator. The order and the number of 
        the simulator actions may be different from what is observed in the 
        sample.
    decagg : function (decstate -> int)
        A function that assigns the number of the aggregate state and its index
        within the aggregation to each individual sample *decision* state.
    noaction : action, optional
        Action index to take in states that do not have a defined policy. The policy 
        is considered to be undefined when it is negative. This only works for 
        deterministic policies.
    
    Returns
    -------
    policy : function (decstate -> action)
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
            #TODO: np.random.choice is particularly slow
            policy = lambda decstate: np.random.choice(actions,1,p=policy_vec[decagg(decstate)])[0]
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
 

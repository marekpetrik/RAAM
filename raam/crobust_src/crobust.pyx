# distutils: language = c++
# distutils: sources = craam/src/RMDP.cpp craam/src/Action.cpp craam/src/definitions.cpp craam/src/State.cpp craam/src/Transition.cpp

import numpy as np 
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.utility cimport pair, tuple
from libcpp cimport bool
import statistics
from collections import namedtuple 
from math import sqrt
import warnings 

cdef extern from "../../craam/include/RMDP.hpp":
    pair[vector[double],double] worstcase_l1(const vector[double] & z, 
                                            const vector[double] & q, double t)

    ctypedef enum SolutionType:
        Robust = 0
        Optimistic = 1 
        Average = 2

    cdef cppclass Solution:
        vector[double] valuefunction
        vector[long] policy
        vector[long] outcomes
        vector[vector[double]] outcome_dists
        double residual
        long iterations

    cdef cppclass Transition:
        vector[long] indices
        vector[double] probabilities
        vector[double] rewards
        double prob_sum

    cdef cppclass RMDP:
        RMDP(int state_count) except +

        void add_transition(long fromid, long actionid, long outcomeid, 
                            long toid, double probability, double reward) except + 

        void set_distribution(long fromid, long actionid, 
                                const vector[double] & distribution, 
                                double threshold) except +
        void set_uniform_distribution(double threshold);
        void set_uniform_thresholds(double threshold) except +

        double get_reward(long stateid, long actionid, long outcomeid, 
                            long sampleid) except +
        void set_reward(long stateid, long actionid, long outcomeid, long sampleid, 
                            double reward) except +
        long sample_count(long stateid, long actionid, long outcomeid) except +

        double get_threshold(long stateid, long actionid) except +
        void set_threshold(long stateid, long actionid, double threshold) except +

        void normalize()

        long state_count() except +
        long action_count(long stateid) except +
        long outcome_count(long stateid, long actionid) except +
        long transitions_count(long stateid, long actionid, long outcomeid) except +
        
        Transition get_transition(long stateid,long actionid,long outcomeid) except +

        Solution vi_gs(vector[double] valuefunction, double discount, unsigned long iterations, double maxresidual, SolutionType type) except +
        Solution vi_gs_l1(vector[double] valuefunction, double discount, unsigned long iterations, double maxresidual, SolutionType type) except +

        Solution vi_jac(const vector[double] & valuefunction, double discount, unsigned long iterations, double maxresidual, SolutionType type) except +
        Solution vi_jac_l1(const vector[double] & valuefunction, double discount, unsigned long iterations, double maxresidual, SolutionType type) except +

        Solution mpi_jac(const vector[double] & valuefunction, double discount, unsigned long politerations, double polmaxresidual, unsigned long valiterations, double valmaxresidual, SolutionType type) except +
        Solution mpi_jac_l1(const vector[double] & valuefunction, double discount, unsigned long politerations, double polmaxresidual, unsigned long valiterations, double valmaxresidual, SolutionType type) except +

        void transitions_to_csv_file(const string & filename, bool header) except +
        string to_string() except +

        void copy_into(RMDP& target) except + 

cpdef cworstcase_l1(np.ndarray[double] z, np.ndarray[double] q, double t):
    """
    o = cworstcase_l1(z,q,t)
    
    Computes the solution of:
    min_p   p^T * z
    s.t.    ||p - q|| <= t
            1^T p = 1
            p >= 0
            
    where o is the objective value
          
    Notes
    -----
    This implementation works in O(n log n) time because of the sort. Using
    quickselect to choose the right quantile would work in O(n) time.
    
    The parameter z may be a masked array. In that case, the distribution values 
    are normalized to the unmasked entries.
    """
    return worstcase_l1(z,q,t).second


# a contained used to hold the dictionaries used to map sample states to MDP states
StateMaps = namedtuple('StateMaps',['decstate2state','expstate2state',\
                                    'decstate2outcome','expstate2outcome'])

cdef class RoMDP:
    """
    Contains the definition of the robust MDP and related optimization algorithms.
    The algorithms can handle both robust and optimistic solutions.
    
    The states, actions, and outcomes are identified by consecutive ids, independently
    numbered for each type.
    
    Initialization requires the number of states.
    
    Parameters
    ----------
    statecount : int
        An estimate of the numeber of states (for pre-allocation). When more states
        are added, the estimate is readjusted.
    discount : double
        The discount factor
    """
    
    cdef RMDP *thisptr
    cdef public double discount

    def __cinit__(self, int statecount, double discount):
        self.thisptr = new RMDP(statecount)
        self.discount = discount        

    def __init__(self, int statecount, double discount):
        self.thisptr = new RMDP(statecount)
        self.discount = discount
        
    def __dealloc__(self):
        del self.thisptr
                
    cpdef add_transition(self, long fromid, long actionid, long outcomeid, long toid, double probability, double reward):
        """
        Adds a single transition sample (robust or non-robust) to the Robust MDP representation.
        
        Parameters
        ----------
        fromid : long
            Unique identifier of the source state of the transition 
        actionid : long
            Identifier of the action. It is unique for the given state
        outcomeid : long
            Identifier of the outcome. It is unique for the given state
        toid : long
            Unique identifier of the target state of the transition
        probability : float
            Probability of the distribution
        reward : float
            Reward associated with the transition
        """
        self.thisptr.add_transition(fromid, actionid, outcomeid, toid, probability, reward)

    cpdef add_transition_nonrobust(self, long fromid, long actionid, long toid, double probability, double reward):
        """
        Adds a single transition sample using outcome with id = 0. This function
        is meant to be used for constructing a non-robust MDP.

        Parameters
        ----------
        fromid : long
            Unique identifier of the source state of the transition 
        toid : long
            Unique identifier of the target state of the transition
        actionid : long
            Identifier of the action. It is unique for the given state
        probability : float
            Probability of the distribution
        reward : float
            Reward associated with the transition
        """        
        self.thisptr.add_transition(fromid, actionid, 0, toid, probability, reward)

    cpdef set_distribution(self, long fromid, long actionid, np.ndarray[double] distribution, double threshold):
        """
        Sets the base distribution over the states and the threshold
        
        Parameters
        ----------
        fromid : int
            Number of the originating state
        actionid : int
            Number of the actions
        distribution : np.ndarray
            Distributions over the outcomes (should be a correct distribution)
        threshold : double
            The difference threshold used when choosing the outcomes
        """
        if abs(np.sum(distribution) - 1) > 0.001:
            raise ValueError('incorrect distribution (does not sum to one)', distribution)
        if np.min(distribution) < 0:
            raise ValueError('incorrect distribution (negative)', distribution)    
        self.thisptr.set_distribution(fromid,actionid,distribution,threshold)
        
    cpdef set_uniform_distributions(self, double threshold):
        """
        Sets all the outcome distributions to be uniform.
        
        Parameters
        ----------
        threshold : double
            The default threshold for the uncertain sets
        """
        self.thisptr.set_uniform_distribution(threshold)

    cpdef set_uniform_thresholds(self, double threshold):
        """
        Sets the same threshold for all states.
        
        Can use ``self.set_distribution`` to set the thresholds individually for 
        each states and action.
        
        See Also
        --------
        self.set_distribution
        """
        self.thisptr.set_uniform_thresholds(threshold)

    cpdef vi_gs(self, int iterations, valuefunction = np.empty(0), \
                            double maxresidual=0, int stype=0):
        """
        Runs value iteration using the worst case (simplex) distribution for the 
        outcomes.
        
        This is the "Gauss-Seidel" kind of value iteration in which the state values
        are updated one at a time and directly used in subsequent iterations.
        
        This version is not parallelized (and likely would be hard to).
        
        Parameters
        ----------
        iterations : int
            Maximal number of iterations
        valuefunction : np.ndarray, optional
            The initial value function. Created automatically if not provided.            
        maxresidual : double, optional
            Maximal residual at which the iterations stop. A negative value
            will ensure the necessary number of iterations.
        stype : int  {0, 1, 2}
            Robust (0) or optimistic (1) solution or (2) average solution. One
            can use e.g. robust.SolutionType.Robust.value.
            
        Returns
        -------
        valuefunction : np.ndarray
            Optimized value function
        policy : np.ndarray
            Policy greedy for value function
        residual : double
            Residual for the value function
        iterations : int
            Number of iterations taken
        outcomes : np.ndarray
            Outcomes selected
        
        See Also
        --------
        SolutionType
        """
        
        if valuefunction.shape[0] == 0:
            # create an appropriate zero initial value function vector
            valuefunction = np.zeros(self.thisptr.state_count(), dtype=np.double)
        elif valuefunction.shape[0] != self.thisptr.state_count():
            raise ValueError('Value function dimensions must match the number of states.')

        cdef SolutionType st;
        if stype == 0:
            st = Robust
        elif stype == 1:
            st = Optimistic
        elif stype == 2:
            st = Average
 
        cdef Solution sol = self.thisptr.vi_gs(valuefunction,self.discount,iterations,maxresidual,st)
        return np.array(sol.valuefunction), np.array(sol.policy), sol.residual, \
                sol.iterations, sol.outcomes
        

    cpdef vi_gs_l1(self, int iterations, valuefunction = np.empty(0), \
                    double maxresidual=0, int stype=0):
        """
        Runs value iteration using the worst distribution constrained by the threshold 
        and l1 norm difference from the base distribution.
        
        This is the "Gauss-Seidel" kind of value iteration in which the state values
        are updated one at a time and directly used in subsequent iterations.
        
        This version is not parallelized (and likely would be hard to).
        
        Parameters
        ----------
        iterations : int
            Maximal number of iterations
        valuefunction : np.ndarray, optional
            The initial value function. Created automatically if not provided.            
        maxresidual : double, optional
            Maximal residual at which the iterations stop. A negative value
            will ensure the necessary number of iterations.
        stype : int  {0, 1, 2}, optional
            Robust (0) or optimistic (1) solution or (2) average solution. One
            can use e.g. robust.SolutionType.Robust.value.
            
        Returns
        -------
        valuefunction : np.ndarray
            Optimized value function
        policy : np.ndarray
            Policy greedy for value function
        residual : double
            Residual for the value function
        iterations : int
            Number of iterations taken
        outcome_dists : np.ndarray[np.ndarray]
            Distributions of outcomes
        """
        
        if valuefunction.shape[0] == 0:
            # create an appropriate zero initial value function vector
            valuefunction = np.zeros(self.thisptr.state_count(), dtype=np.double)
        elif valuefunction.shape[0] != self.thisptr.state_count():
            raise ValueError('Value function dimensions must match the number of states.')

        cdef SolutionType st;
        if stype == 0:
            st = Robust
        elif stype == 1:
            st = Optimistic
        elif stype == 2:
            st = Average
 
        cdef Solution sol = self.thisptr.vi_gs_l1(valuefunction,self.discount,iterations,maxresidual,st)
        
        return np.array(sol.valuefunction), np.array(sol.policy), sol.residual, \
                sol.iterations, sol.outcome_dists
        
    cpdef vi_jac(self, int iterations,valuefunction = np.empty(0), \
                                    double maxresidual=0, int stype=0):
        """
        Runs value iteration using the worst case (simplex) distribution for the 
        outcomes.
        
        This is the parallel version of the update with values updates for all states
        simultaneously.
        
        Parameters
        ----------
        iterations : int
            Maximal number of iterations
        valuefunction : np.ndarray, optional
            The initial value function. Created automatically if not provided.            
        maxresidual : double, optional
            Maximal residual at which the iterations stop. A negative value
            will ensure the necessary number of iterations.
        stype : int  (0, 1, 2}
            Robust (0) or optimistic (1) solution or (2) average. One
            can use e.g. robust.SolutionType.Robust.value.
            
        Returns
        -------
        valuefunction : np.ndarray
            Optimized value function
        policy : np.ndarray
            Policy greedy for value function
        residual : double
            Residual for the value function
        iterations : int
            Number of iterations taken
        outcomes : np.ndarray
            Outcomes selected
        """
        
        if valuefunction.shape[0] == 0:
            # create an appropriate zero initial value function vector
            valuefunction = np.zeros(self.thisptr.state_count(), dtype=np.double)
        elif valuefunction.shape[0] != self.thisptr.state_count():
            raise ValueError('Value function dimensions must match the number of states.')        
                
        cdef SolutionType st;
        if stype == 0:
            st = Robust
        elif stype == 1:
            st = Optimistic
        elif stype == 2:
            st = Average

        cdef Solution sol = self.thisptr.vi_jac(valuefunction,self.discount,iterations,maxresidual,st)
        return np.array(sol.valuefunction), np.array(sol.policy), sol.residual, \
                sol.iterations, sol.outcomes

    cpdef vi_jac_l1(self, long iterations, valuefunction = np.empty(0), \
                                    double maxresidual = 0, int stype=0):
        """
        Runs value iteration using the worst distribution constrained by the threshold 
        and l1 norm difference from the base distribution.
        
        This is the parallel version of the update with values updates for all states
        simultaneously.
        
        Parameters
        ----------
        iterations : int
            Maximal number of iterations
        valuefunction : np.ndarray, optional
            The initial value function. Created automatically if not provided.            
        maxresidual : double
            Maximal residual at which the iterations stop. A negative value
            will ensure the necessary number of iterations.
        stype : int  (0, 1, 2}
            Robust (0) or optimistic (1) solution or (2) average solution. One
            can use e.g. robust.SolutionType.Robust.value.
            
        Returns
        -------
        valuefunction : np.ndarray
            Optimized value function
        policy : np.ndarray
            Policy greedy for value function
        residual : double
            Residual for the value function
        iterations : int
            Number of iterations taken
        outcome_dists : np.ndarray[np.ndarray]
            Distributions of outcomes
        """
        
        if valuefunction.shape[0] == 0:
            # create an appropriate zero initial value function vector
            valuefunction = np.zeros(self.thisptr.state_count(), dtype=np.double)
        elif valuefunction.shape[0] != self.thisptr.state_count():
            raise ValueError('Value function dimensions must match the number of states.')

        cdef SolutionType st;
        if stype == 0:
            st = Robust
        elif stype == 1:
            st = Optimistic
        elif stype == 2:
            st = Average

        cdef Solution sol = self.thisptr.vi_jac_l1(valuefunction,self.discount,iterations,maxresidual,st)
        return np.array(sol.valuefunction), np.array(sol.policy), sol.residual, \
                sol.iterations, sol.outcome_dists

    cpdef mpi_jac(self, long iterations, valuefunction = np.empty(0), \
                                    double maxresidual = 0, long valiterations = 1000, int stype=0):
        """
        Runs modified policy iteration using the worst distribution constrained by the threshold 
        and l1 norm difference from the base distribution.
        
        This is the parallel version of the update with values updates for all states
        simultaneously.
        
        Parameters
        ----------
        iterations : int
            Maximal number of iterations
        valuefunction : np.ndarray, optional
            The initial value function. Created automatically if not provided.            
        maxresidual : double
            Maximal residual at which the iterations stop. A negative value
            will ensure the necessary number of iterations.
        valiterations : int, optional
            Maximal number of iterations for value function computation
        stype : int  (0, 1, 2}
            Robust (0) or optimistic (1) solution or (2) average solution. One
            can use e.g. robust.SolutionType.Robust.value.
            
        Returns
        -------
        valuefunction : np.ndarray
            Optimized value function
        policy : np.ndarray
            Policy greedy for value function
        residual : double
            Residual for the value function
        iterations : int
            Number of iterations taken
        outcomes : np.ndarray
            Outcomes selected
        """
        
        if valuefunction.shape[0] == 0:
            # create an appropriate zero initial value function vector
            valuefunction = np.zeros(self.thisptr.state_count(), dtype=np.double)
        elif valuefunction.shape[0] != self.thisptr.state_count():
            raise ValueError('Value function dimensions must match the number of states.')

        # TODO: what it the best value to use here?
        cdef double valresidual = maxresidual / 2

        cdef SolutionType st;
        if stype == 0:
            st = Robust
        elif stype == 1:
            st = Optimistic
        elif stype == 2:
            st = Average

        cdef Solution sol = self.thisptr.mpi_jac(valuefunction,self.discount,iterations,maxresidual,\
                                                        valiterations, valresidual,st)
        return np.array(sol.valuefunction), np.array(sol.policy), sol.residual, \
                sol.iterations, sol.outcomes


    cpdef mpi_jac_l1(self, long iterations, valuefunction = np.empty(0), \
                        long valiterations = 1000, double maxresidual = 0, int stype=0):
        """
        Runs modified policy iteration using the worst distribution constrained by the threshold 
        and l1 norm difference from the base distribution.
        
        This is the parallel version of the update with values updates for all states
        simultaneously.
        
        Parameters
        ----------
        iterations : int
            Maximal number of iterations
        valuefunction : np.ndarray, optional
            The initial value function. Created automatically if not provided.            
        maxresidual : double
            Maximal residual at which the iterations stop. A negative value
            will ensure the necessary number of iterations.
        valiterations : int, optional
            Maximal number of iterations for value function computation
        stype : int  (0, 1, 2}
            Robust (0) or optimistic (1) solution or (2) average solution. One
            can use e.g. robust.SolutionType.Robust.value.
            
        Returns
        -------
        valuefunction : np.ndarray
            Optimized value function
        policy : np.ndarray
            Policy greedy for value function
        residual : double
            Residual for the value function
        iterations : int
            Number of iterations taken
        outcome_dists : np.ndarray[np.ndarray]
            Distributions of outcomes
        """
        
        if valuefunction.shape[0] == 0:
            # create an appropriate zero initial value function vector
            valuefunction = np.zeros(self.thisptr.state_count(), dtype=np.double)
        elif valuefunction.shape[0] != self.thisptr.state_count():
            raise ValueError('Value function dimensions must match the number of states.')

        # TODO: what it the best value to use here?
        cdef double valresidual = maxresidual / 2

        cdef SolutionType st;
        if stype == 0:
            st = Robust
        elif stype == 1:
            st = Optimistic
        elif stype == 2:
            st = Average

        cdef Solution sol = self.thisptr.mpi_jac_l1(valuefunction,self.discount,iterations,maxresidual,\
                                                    valiterations, valresidual, st)
                                                    

        return np.array(sol.valuefunction), np.array(sol.policy), sol.residual,\
                sol.iterations, sol.outcome_dists

    cpdef from_sample_matrices(self, dectoexp, exptodec, actions, rewards):
        """
        Add samples from matrices generated by raam.robust.matrices.

        Note: The base distributions over the outcomes are assumed to be uniform.
        
        Parameters
        ----------
        dectoexp : numpy.ndarray
            List of transitions for all aggregate states. For each aggregate state
            the list contains a *masked* matrix (np.ndarray) with dimensions 
            #actions x #outcomes. The entries with no samples are considered to be masked.
            Each outcome corresponds to a decision state (from decision samples)
            with a unique index. Each entry is an index of the corresponding
            expectation state (row number in exptodec).
        exptodec : scipy.sparse.dok_matrix
            A sparse transition matrix from expectation states to decision states.
        actions : list
            List of actions available in the problem. Actions are sorted alphabetically.
        rewards : numpy.ndarray
            Average reward for each expectation state
        """
        cdef int actioncount = len(actions)
        cdef int statecount = dectoexp.shape[0]
        cdef int s,a,o,ns

        for s in range(statecount):
            actionoutcomes = dectoexp[s]
            for a in range(actioncount):
                if actionoutcomes is None:
                    continue
                outcomecount = actionoutcomes.shape[1]
                realoutcomecount = 0
                for o in range(outcomecount):
                    if actionoutcomes.mask[a,o]:
                        continue
                    es = exptodec[actionoutcomes[a,o],:].tocoo()
                    rew = rewards[actionoutcomes[a,o]]
                    
                    es_size = es.col.shape[0]
                    cols = es.col
                    data = es.data
                    
                    for ns in range(es_size):
                        toid = cols[ns]
                        prob = data[ns]
                        self.add_transition(s,a,realoutcomecount,toid,prob,rew)
                    realoutcomecount += 1
                
                if(realoutcomecount > 0):
                    dist = np.ones(realoutcomecount) / realoutcomecount
                    self.set_distribution(s,a,dist,2)

    cpdef from_matrices(self, np.ndarray[double,ndim=3] transitions, np.ndarray[double,ndim=2] rewards, \
        np.ndarray[long] actions, np.ndarray[long] outcomes, double ignorethreshold = 1e-10):
        """
        Constructs an MDP from transition matrices. The function is meant to be
        called only once and cannot be used to re-initialize the transition 
        probabilities.
        
        Number of states is ``n = |states|``. The number of available action-outcome
        pairs is ``m``.
        
        Parameters
        ----------
        transitions : np.ndarray[double,double,double] (n x n x m)
            Each row represents a transition for the specified state, the
            third dimension is used to represent the transition probabilities
            for specific outcomes and actions.
        rewards : np.ndarray[double, double] (n x m)
            The rewards for each state and action
        actions : np.ndarray[long] (m)
            The id of the action for the state
        outcomes : np.ndarray[long] (m)
            The id of the outcome for the state
        ignorethreshold : double, optional
            Any transition probability less than the threshold is ignored leading to 
            sparse representations. If not provided, no transitions are ignored
        """
        cdef int actioncount = len(actions) # really the number of action-outcome pairs
        cdef int statecount = transitions.shape[0]

        if actioncount != len(outcomes):
            raise ValueError('Length of actions and outcomes must match.')
        if actioncount != transitions.shape[2] or actioncount != rewards.shape[1]:
            raise ValueError('The number of actions must match the 3rd dimension of transitions and the 2nd dimension of rewards.')
        if statecount != transitions.shape[1] or statecount != rewards.shape[0]:
            raise ValueError('The number of states in transitions and rewards is inconsistent.')
        if len(set(zip(actions,outcomes))) != actioncount:
            raise ValueError('The action and outcome pairs must be unique.')

        cdef int aoindex, fromid, toid
        cdef int actionid, outcomeid
        cdef double transitionprob, rewardval

        for aoindex in range(actioncount):    
            for fromid in range(statecount):
                for toid in range(statecount):
                    actionid = actions[aoindex]
                    outcomeid = outcomes[aoindex]
                    transitionprob = transitions[fromid,toid,aoindex]
                    if transitionprob <= ignorethreshold:
                        continue
                    rewardval = rewards[fromid,aoindex]
                    self.add_transition(fromid, actionid, outcomeid, toid, transitionprob, rewardval)


    cpdef long state_count(self):
        """
        Returns the number of states
        """
        return self.thisptr.state_count()
        
    cpdef long action_count(self, long stateid):
        """
        Returns the number of actions
        
        Parameters
        ----------
        stateid : int
            Number of the state
        """
        return self.thisptr.action_count(stateid)
        
    cpdef long outcome_count(self, long stateid, long actionid):
        """
        Returns the number of outcomes
        
        Parameters
        ----------
        stateid : int
            Number of the state
        actionid : int
            Number of the action
        """
        return self.thisptr.outcome_count(stateid, actionid)

    cpdef double get_reward(self, long stateid, long actionid, long outcomeid, long sampleid):
        """
        Returns the reward for the given state, action, and outcome
        """
        return self.thisptr.get_reward(stateid, actionid, outcomeid, sampleid)
    
    cpdef set_reward(self, long stateid, long actionid, long outcomeid, long sampleid, double reward):
        """
        Sets the reward for the given state, action, outcome, and sample
        """
        self.thisptr.set_reward(stateid, actionid, outcomeid, sampleid, reward)
        
    cpdef long sample_count(self, long stateid, long actionid, long outcomeid):
        """
        Returns the number of samples (single-state transitions) for the action and outcome
        """
        return self.thisptr.sample_count(stateid, actionid, outcomeid)

    def list_samples(self):
        """
        Returns a list of all samples in the problem. Can be useful for debugging.
        """
        cdef long stateid, actionid, outcomeid
        cdef Transition tran
        cdef long tcount, tid
        
        result = [('state','action','outcome','tostate','prob','rew')]
        
        for stateid in range(self.state_count()):
            for actionid in range(self.action_count(stateid)):
                for outcomeid in range(self.outcome_count(stateid,actionid)):
                    tran = self.thisptr.get_transition(stateid,actionid,outcomeid)
                    tcount = len(tran.indices)
                    for tid in range(tcount):
                        result.append( (stateid,actionid,outcomeid,\
                            tran.indices[tid], tran.probabilities[tid], tran.rewards[tid]) )

        return result

    cpdef copy(self):
        """
        Makes a copy of the object
        """
        r = RoMDP(0, self.discount)
        self.thisptr.copy_into( r.thisptr[0] ) 
        return r

    cpdef double get_threshold(self, long stateid, long actionid):
        """ Returns the robust threshold for a given state """
        return self.thisptr.get_threshold(stateid, actionid)
    
    cpdef set_threshold(self, long stateid, long actionid, double threshold):
        """ Sets the robust threshold for a given state """
        self.thisptr.set_threshold(stateid, actionid, threshold)
        
    cpdef transitions_to_csv_file(self, filename, header = True):
        """ Saves the transitions to a csv file """
        self.thisptr.transitions_to_csv_file(filename, header)

    cpdef string to_string(self):
        cdef string result = self.thisptr.to_string()
        return result 

    cpdef normalize(self):
        self.thisptr.normalize()

    def __dealloc__(self):
        del self.thisptr


class SRoMDP:
    """
    Robust MDP constructed from samples and an aggregation.
    
    See :method:`from_samples` for the description of basic usage.
    
    Parameters
    ----------
    states : int
        Initial number of states. State space is automaticaly expanded when more
        samples become available.
    discount : float
        Discount factor used in the MDP.
    """
    
    def __init__(self,states,discount):
        # state mappings
        self.rmdp = RoMDP(states,sqrt(discount))
        self.discount = discount
        
        # decstate2state, expstate2state, decstate2outcome, expstate2outcome
        self.statemaps = StateMaps({},{},{},{})
        
        # the following dictionaries are used in order to properly weigh samples 
        # when added multiple times
                        
        # decision states
        self.dcount_sao = {} # maps state,action,outcome to the number of observations
            # this is uses to combine multiple samples
        
        # expectation states
        self.ecount_sao = {} # maps state,action,outcome to the number of observations
            # this is uses to combine multiple samples
    
    def from_samples(self, samples, decagg_big, decagg_small, \
                        expagg_big, expagg_small, actagg):
        """
        Loads data to the MDP from the provided samples given aggregation functions.
        Each decision state that belongs to a single aggregated state corresponds 
        to an (wost-case) outcome. The function does not return anything.
        
        Both expectation and decision states are translated to separate RMDP states
        
        Important: the discount factor used internally with the RoMDP must 
        be sqrt(discount) for to behave as discount; this correction is handled 
        automatically by the class.
        
        Parameters
        ----------
        samples : raam.Samples
            List of samples
        decagg_big : function
            Aggregation function for decision states, used to construct the 
            actual aggregation. The function should return an integer 
            (could be negative).
        decagg_small : function
            Aggregation function used to construct outcomes and the value is 
            relative to the state given by ``decagg_big``. The solution is 
            computed as the worst-case over these outcomes. This can be just a
            finer aggregation function than decagg_big, or could come from 
            multiple runs. The function should return an integer 
            (could be negative). Use ``features.IdCache`` to simply use the state
            identity.
        expagg_big : function
            Aggregation function for expectation states. This is used to average
            the transition probabilities. The function should return an integer 
            (could be negative). Use features.IdCache to simply use the state
            identity.
        expagg_small : function
            Aggregation function used to construct outcomes for expectation states.
            The function can be ``None`` which means that all expectation states
            are aggregated into one.
        actagg : function
            Aggregation function for actions. The function should return an integer 
            (could be negative).
        
        Note
        ----
        If the aggregation functions return a floating point then the number is
        rounded to an integer and used as the index of the state or the action. 
        
        See Also
        --------
        decvalue
        """
        cdef long aggds_big, aggds_small, agges_big, agges_small
        cdef long numdecstate, numexpstate, numaction, numoutcome
        cdef long mdpstates = self.rmdp.state_count()
        cdef RoMDP rmdp = self.rmdp        
        
        # if there is no small aggregation provided, then just assume that there 
        # is no aggregation
        if expagg_small is None:
            expagg_small = lambda x: 0
        
        # maps decision states (aggregated) to the states of the RMDP
        decstate2state = self.statemaps.decstate2state
        # maps expectation states (aggregated) to the states of the RMDP
        expstate2state = self.statemaps.expstate2state
        # maps decision states (small aggregation) to dictionaries of outcomes in the MDP
        decstate2outcome = self.statemaps.decstate2outcome
        # maps expectation states (small aggregated) to  outcomes in the MDP
        expstate2outcome = self.statemaps.expstate2outcome
        
        dcount_sao_old = self.dcount_sao.copy()
        ecount_sao_old = self.ecount_sao.copy()
        
        # *** process decision samples
        for ds in samples.decsamples():
            
            # compute the mdp state for the decision state
            aggds_big = decagg_big(ds.decStateFrom)
            if aggds_big in decstate2state:
                numdecstate = decstate2state[aggds_big]
            else:
                numdecstate = mdpstates
                mdpstates += 1
                decstate2state[aggds_big] = numdecstate
            
            # compute the mdp state for the expectation state
            agges_big = expagg_big(ds.expStateTo)
            if agges_big in expstate2state:
                numexpstate = expstate2state[agges_big]
            else:
                numexpstate = mdpstates
                mdpstates += 1
                expstate2state[agges_big] = numexpstate
            
            # compute action aggregation, the mapping is 1->1
            numaction = actagg(ds.action)
            
            # compute the outcome aggregation
            aggds_small = decagg_small(ds.decStateFrom)
            
            outcomedict = decstate2outcome.get((aggds_big,numaction), None)
            if outcomedict is None:
                outcomedict = {}
                decstate2outcome[(aggds_big,numaction)] = outcomedict
            
            if aggds_small in outcomedict:
                numoutcome = outcomedict[aggds_small]
            else:
                numoutcome = len(outcomedict)
                outcomedict[aggds_small] = numoutcome

            # update the counts for the sample
            sao = (numdecstate,numaction,numoutcome)
            self.dcount_sao[sao] = self.dcount_sao.get(sao,0) + 1
            
            # now, just add the transition
            # use the old counts to compute the weight in order for the normalization
            # to work
            weight = 1.0 / float(dcount_sao_old.get(sao,1))
            rmdp.add_transition(numdecstate,numaction,numoutcome,numexpstate,weight,0.0)
            
        # *** process expectation samples
        # one action and outcome per state
        for es in samples.expsamples():
            # compute the mdp state for the expectation state
            agges_big = expagg_big(es.expStateFrom)
            if agges_big in expstate2state:
                numexpstate = expstate2state[agges_big]
            else:
                numexpstate = mdpstates
                mdpstates += 1
                expstate2state[agges_big] = numexpstate

            # compute the mdp state for the decision state
            aggds_big = decagg_big(es.decStateTo)
            if aggds_big in decstate2state:
                numdecstate = decstate2state[aggds_big]
            else:
                numdecstate = mdpstates
                mdpstates += 1
                decstate2state[aggds_big] = numdecstate
            
            # compute action aggregation
            numaction = 0   # only one action
            
            # compute the outcome aggregation
            agges_small = expagg_small(es.expStateFrom)
            
            outcomedict = expstate2outcome.get(agges_big, None)
            
            if outcomedict is None:
                outcomedict = {}
                expstate2outcome[agges_big] = outcomedict
                
            if agges_small in outcomedict:
                numoutcome = outcomedict[agges_small]
            else:
                numoutcome = len(outcomedict)
                outcomedict[agges_small] = numoutcome

            # update the counts for the sample
            so = (numexpstate,numoutcome)
            self.ecount_sao[so] = self.ecount_sao.get(so,0) + 1
            
            # now, just add the transition
            # use the old counts to compute the weight in order for the normalization
            # to work
            weight = 1.0 / float(ecount_sao_old.get(so,1))    
                
            # now, just add the transition
            self.rmdp.add_transition(numexpstate,numaction,numoutcome,numdecstate,\
                            weight*es.weight,es.reward)
            
        # add a transition to a bad state for state-actions with no outcomes
        # these state-action pairs are created automatically 
        cdef double bad = -float('inf')
        cdef long numbadstate = self.rmdp.state_count()
        self.rmdp.add_transition(numbadstate,0,0,numbadstate,1.0,0)
        
        for numstate in range(self.rmdp.state_count()):
            for numaction in range(self.rmdp.action_count(numstate)):
                if self.rmdp.outcome_count(numstate,numaction) == 0:
                    self.rmdp.add_transition(numstate,numaction,0,numbadstate,1.0,bad)
        
        # normalize transition weights
        self.rmdp.normalize()
                
    def decvalue(self,states,value,minstate=0):
        """
        Corrects the value function and maps the result from an algorithm
        to the value function for decision states.
        
        The function also corrects for the discrepancy in the discount factor,
        which is applied twice. 
        
        Parameters
        ----------
        states : int
            Number of states for which the express the value function. The states
            must be numbered from 0 to states - 1
        value : numpy.array
            Value function array as an output from the optimization methods. This
            uses an internal representation.
        minstate : int, optional
            The minimal index of the state. The default is 0.
        
        Returns
        -------
        out : numpy.array
            Value function for decision states
        """
        cdef long i, index
        cdef double discountadj = 1/sqrt(self.discount)
        result = np.empty((states,))
        for i in range(minstate,minstate+states):
            index = self.statemaps.decstate2state.get(i,-1)
            if index >= 0:
                result[minstate+i] = value[index] * discountadj
            else:
                result[minstate+i] = float('nan')
        return result

    def expvalue(self,states,value,minstate=0):
        """
        Corrects the value function and maps the result from an algorithm
        to the value function for expectation states.
        
        Parameters
        ----------
        states : int
            Number of states for which the express the value function. The states
            must be numbered from 0 to states - 1
        value : numpy.array
            Value function array as an output from the optimization methods. This
            uses an internal representation.
        minstate : int, optional
            The minimal index of the state. The default is 0.
        
        Returns
        -------
        out : numpy.array
            Value function for decision states
        """
        cdef long i, index
        result = np.empty((states,))
        for i in range(minstate,minstate+states):
            index = self.statemaps.expstate2state.get(i,-1)
            if index >= 0:
                result[minstate+i] = value[index] 
            else:
                result[minstate+i] = float('nan')
        return result
                
    def decpolicy(self,states,policy,minstate=0):
        """
        Corrects the policy function (a vector) and maps the result from an algorithm
        to the policy over decision states.
        
        The function also corrects for the discrepancy in the discount factor,
        which is applied twice. 
        
        Parameters
        ----------
        states : int
            Number of states for which the express the policy function. The states
            must be numbered from 0 to states - 1
        policy : numpy.array
            Value function array as an output from the optimization methods. This
            uses an internal representation.
        minstate : int, optional
            The minimal index of the state. The default is 0.

        Returns
        -------
        out : numpy.array(int)
            Policy function for decision states
        """
        cdef long i, index
        result = np.empty((states,),dtype=int)
        for i in range(minstate,minstate+states):
            index = self.statemaps.decstate2state.get(i,-1)
            if index >= 0:
                result[minstate+i] = policy[index]
            else:
                result[minstate+i] = -1
        return result

    def statemaps(self):
        """
        Returns the maps from sample states to actual MDP states
        """
        return self.statemaps        
        
    def samplecount(self):
        """
        Returns the number of samples in the object
        """
        return sum(self.dcount_sao.values()) + sum(self.ecount_sao.values())

    def expstate_numbers(self):
        """
        Returns numbers of the internal RoMDP states that correspond to the
        expectation states as well as the expectation state numbers.
        
        Returns
        -------
        expstate_original : list
            Original numbers assigned to the expectation states
        expstate_index : list        
            Index of the expectation state in the constructed RMDP
        """
        return list( zip(*self.statemaps.expstate2state.items()) )
        
    def decstate_numbers(self):
        """
        Returns numbers of the internal RoMDP states that correspond to the
        decision states as well as the decision state numbers.

        Returns
        -------
        decstate_original : list
            Original numbers assigned to the decision states
        decstate_index : list        
            Index of the decision state in the constructed RMDP
        """
        
        return list( zip(*self.statemaps.decstate2state.items()) )
"""
================================================================
Plain python implementation of robust MDPs (:mod:`raam.robust`)
================================================================

This implementation is slow, incomplete, and not as thoroughly
tested as the C++ version. Use at your own risk.
"""
import raam
import itertools
from scipy import sparse
import numpy as np
from numpy import ma
from enum import Enum

class SolutionType(Enum):
    """
    Type of the solution to seek
    """
    Robust = 0
    Optimistic = 1
    Average = 2

def __agg__(s):
    """ Returns the first component of the list. """
    r = int(s[0])
    assert(r >= 0)
    return int(s[0])

def __ind__(s,add=False):
    """ Returns the first two components of the list"""    
    return (int(s[0]), int(s[1]))

def worstcase_l1(z,q,t):
    """
    o = worstcase_l1(z,q,t)
    
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
    if t > 2 or t < 0:
        raise ValueError('Bound t must be in [0,2].')    
    if q.dtype != np.float:
        raise ValueError('Value q must be a flot array.')
    
    # if it is a masked array, then simply remove the masked entries
    if np.ma.is_masked(z):
        mask = np.ma.getmaskarray(z)
        q = q[~mask]
        z = z[~mask]
        masksum = np.sum(q)
        if masksum <= 0:
            raise ValueError('Unmasked entries must have non-zero sum.')
        q = q / masksum

    # sort items increasingly
    smallest = np.argsort(z)
    
    k = smallest[0]
    epsilon = min( (t/2, 1-q[k]) )
    o = np.copy(q)
    o[k] += epsilon
        
    i = len(smallest) - 1
    while epsilon > 0:        
        k = smallest[i]
        diff = min( (epsilon, o[k]) )
        o[k] -= diff
        epsilon -= diff
        i -= 1

    return np.dot(o,z)
  
def matrices(samples, decagg=__ind__, expagg=__agg__, \
            aggdecstates=None,aggexpstates=None,numoutcomes=None):
    """
    Computes transition matrices from samples that are suitable for use with an 
    aggregation and robust or fragile (optimistic) optimization. Aggregated
    decision states are treated individually (as long as they are assigned
    different indices) by constructing appropriate decision matrices. 
    Aggregated expectation states are treated as identical.
    
    Parameters
    ----------
    samples : raam.samples.Samples
    decagg : function (decstate -> (int,int)), optional.
        A function that assigns the number of the aggregate state and its index
        within the aggregation to each individual sample *decision* state.
        If omitted, the numbers are assigned automatically 
        based on the first and the second feature.
    expagg : function (expstate -> int), optional
        A function that assigns the number of the aggregate state to each individual
        sample *expectation* state. If omitted, the numbers are assigned automatically 
        based on the first feature value. If the state is not meant to be aggregated
        then the aggregation function may return a value that is outside of the
        range (e.g. -1). However, this is only valid if the expectation state is
        the target in decision samples and not a source in expectation samples.
    aggdecstates : int, optional
        Number of aggregated *decision* states. If omitted, the samples are
        traversed to determine the number of aggregations.
    aggexpstates : int, optional
        Number of aggregated *expectation* states. If omitted, the sample are
        traversed to determine the number of aggregations. 
    numoutcomes : list, optional
        Number of outcomes (or indices) for every aggregation states. If omitted,
        it is computed by traversing all samples.

    Returns
    -------
    dectoexp : numpy.ndarray
        List of transitions for all aggregate states. For each aggregate state
        the list contains a *masked* matrix (np.ndarray) with dimensions 
        #actions x #outcomes. The entries with no samples are considered to be masked.
        Each outcome corresponds to a decision state (from decision samples)
        with a unique index. Each entry is an index of the corresponding
        expectation state (row number in exptodec).
    exptodec : scipy.sparse.dok_matrix
        Sparse transition matrix from expectation states to decision states.
    actions : list
        List of actions available in the problem. Actions are sorted alphabetically.
    initial : numpy.ndarray
        Initial distribution
    rewards : numpy.ndarray
        Average reward for each expectation state
        
    Notes
    -----
    If there are no transitions out of an aggregated *decision state* then 
    it is considered to be a terminal state.
    
    If there is no transition out of an expectation state, the transition to 
    this expectation state is ignored.
    
    The values are returned in a dictionary.
    
    Examples
    --------
    >>> samples = examples.chain.simple_samples(7,1)
    >>> result = raam.robust.matrices(samples,decagg=IndexCache(),expagg=IdCache())
    
    See Also
    --------
    IdCache, IdIndexCache
    """
    
    if aggdecstates is None:
        alldecstates = itertools.chain( \
            (decagg(s.decStateFrom)[0] for s in samples.decsamples()), \
            (decagg(s.decStateTo)[0] for s in samples.expsamples())) 
        aggdecstates = max(alldecstates) + 1
        del alldecstates
    
    if aggexpstates is None:
        allexpstates = (expagg(s.expStateFrom) for s in samples.expsamples())
        aggexpstates = max(allexpstates) + 1
        del allexpstates
    
    # Compute expectation state transition probabilities
    exptodec = sparse.lil_matrix( (aggexpstates, aggdecstates) )
    rewards = np.zeros( aggexpstates )

    for e in samples.expsamples():
        eaggnum = expagg(e.expStateFrom)
        if eaggnum < 0 or eaggnum >= aggexpstates:
            continue
        exptodec[eaggnum, decagg(e.decStateTo)[0] ] += e.weight
        rewards[eaggnum] += e.weight * e.reward
    exptodec = exptodec.tocsr()

    normalize = 1 / np.array(exptodec.sum(1).flat)
    exptodec = sparse.diags(normalize, 0).dot(exptodec)
    rewards = (normalize * rewards)
    
    # Compute decision transitions 
    actions = sorted(set(d.action for d in samples.decsamples()))
    actid = { a:i for i,a in enumerate(actions) }
    
    if numoutcomes is None:
        numoutcomes = np.zeros(aggdecstates)
        for s in samples.decsamples():
            aggnumber,index = decagg(s.decStateFrom)
            numoutcomes[aggnumber] = max(numoutcomes[aggnumber], index + 1) 
    elif len(numoutcomes) != aggdecstates:
        raise ValueError('The number of elements of numoutcomes must be the same as the number of aggregated states: %d' % aggdecstates)
    
    # Compute decision to expectation transitions
    dectoexp = np.ndarray(shape=aggdecstates,dtype=np.ndarray)
    for i in range(aggdecstates):
        if numoutcomes[i] > 0:
            # if there are not out-going edges then this is a terminal state
            dectoexp[i] = ma.masked_all( (len(actions), numoutcomes[i] ) , dtype=np.int)
        
    # initialized decision to expectation states
    for s in samples.decsamples():
        dstatefrom = s.decStateFrom
        estateto = s.expStateTo
        action = actid[s.action]
        aggiddstate,indexdstate = decagg(dstatefrom)
        aggidestate = expagg(estateto)
        
        assert dectoexp[aggiddstate][action,indexdstate] == aggidestate or \
                dectoexp[aggiddstate][action,indexdstate] is ma.masked
        
        dectoexp[aggiddstate][action,indexdstate] = aggidestate
    
    # Compute initial distribution
    initial = np.zeros(aggdecstates)
    for d in samples.decsamples():
        if d.step == 0:
            initial[ decagg(d.decStateFrom)[0] ] += 1
    initial = initial / initial.sum(0)

    return {'dectoexp':dectoexp,
            'exptodec':exptodec,
            'actions':actions,
            'initial':initial,
            'rewards':rewards}

def vi_gs(dectoexp,exptodec,rewards,discount,optsteps,updsteps=1,type='robust', \
            uncertainty='simplex',distributions=None,bounds=None):
    """ 
    Runs value iteration for robust or optimistic Markov decision problem. The
    the specification comes from :mod:`raam.robust.matrices`.
    
    Parameters
    ----------
    dectoexp : numpy.ndarray
        List of transitions for all aggregate states. For each aggregate state
        the list contains a *masked* matrix (np.ndarray) with dimensions 
        #actions x #outcomes. The entries with no samples are considered to be masked.
        Each outcome corresponds to a decision state (from decision samples)
        with a unique index. Each entry is an index of the corresponding
        expectation state (row number in exptodec).
        
        An entry that is None is assumed to be a terminal state.
    exptodec : scipy.sparse.matrix
        Sparse transition matrix from expectation states to decision states
    rewards : numpy.ndarray
        Average reward for each expectation state    
    discount : float, array(__getitem__)
        The discount factor in [0,1]. Guaranteed to work when discount < 1, but 
        may also work when discount = 1
    type : {``'robust'``,``'optimistic'``}, optional
        - If ``robust``, the method computes the robust result 
            (with the worst case over outcomes).
        - If ``optimistic``, the method computes the optimistic result
            (with the best case over outcomes).
    uncertainty : {``'simplex'``,``'l1'``}
        The type of uncertainty sets
    distributions : numpy.ndarray (#states) , optional
        Base distributions for each state (action independent). Each entry is 
        a distribution over robust actions.
    bounds : numpy.ndarray (#states)
        Bounds on the deviation from the underlying probability distribution
            
    Returns
    -------
    value : numpy.ndarray (#decstates)
        The computed value function
    policy_vec : numpy.ndarray (#decstates x #actions)
        Probability of actions for each decision state
    bounds : float
        The residual after the computation * 2 / (1-discount). 
        This is the bound on the suboptimality of the solution.
        
    Notes
    -----
    The returned policy is a vector of actions for the aggregated states. Use
    :func:`vec2policy` to create an actual policy.
    
    The value iteration ignores robust samples that have not been sampled and
    also ignores actions that have no samples.
    
    See Also
    --------
    vec2policy
    """
    
    if type == 'robust':
        robust = True
    elif type == 'optimistic':
        robust = False
    else:
        raise ValueError('Type must be in the set {"robust","optimistic"}.')

    statecount = len(dectoexp)
    actioncount = dectoexp[0].shape[0]

    # test and set discounts
    if hasattr(discount,'__getitem__'):
        discounts = discount
        if np.min(discounts) < 0 or np.max(discounts) > 1:
            raise ValueError('Discount factor must be in [0,1]')
        if len(discounts) != statecount:
            raise ValueError('When discounts are an array, the length must match the number of states.')
    else:
        if discount < 0 or discount > 1:
            raise ValueError('Discount factor must be in [0,1]')
        discounts = np.empty(statecount)
        discounts.fill(discount)
        
    if uncertainty == 'simplex':
        pass
    elif uncertainty == 'l1':
        if distributions is None or bounds is None:
            raise ValueError('Bounds and distributions must be provided with l1 norm uncertainty sets.')
    else:
        raise ValueError('Unknown uncertainty set')
    
    values = np.zeros(statecount)
    newvalues = np.zeros(statecount)
    policy = np.zeros(statecount)   # the chosen actions
    scheme = np.zeros(statecount)   # the chosen outcomes
    
    # iterate over states
    for i in range(optsteps):
        for stateindex in range(statecount):
            aomatrix = dectoexp[stateindex]
            
            # if there are no actions to be taken, assume that this is a terminal
            # state and leave the value to be 0
            if aomatrix is None:
                continue
            
            aovalues = ma.masked_all(aomatrix.shape,dtype=float)
            activeindices = np.where((~aomatrix.mask).flat)[0]
            expactiveindices = aomatrix.compressed()
            
            aovalues.put(activeindices, \
                discounts[stateindex] * exptodec[expactiveindices,:].dot(values) + rewards[expactiveindices] )

            if uncertainty == 'simplex':
                if robust:
                    outcomes = aovalues.argmin(1)
                else:
                    outcomes = aovalues.argmax(1)
                
                avalues = aovalues.flat[ \
                    np.ravel_multi_index( (np.arange(len(outcomes)),outcomes), aovalues.shape)]
                                                            
                action = avalues.argmax()
                outcome = outcomes[action]
                
                newvalues[stateindex] = aovalues[action,outcome]

            elif uncertainty == 'l1':
                avalues = np.empty(aovalues.shape[0],dtype=np.float)
                outcomes = np.empty(aovalues.shape[0],dtype=np.float)
                outcomes.fill(1000000)      # the large value is supposed to generate an overflow
                
                dist = distributions[stateindex]
                bound = bounds[stateindex]
                
                if robust:
                    for index,outcomevalues in enumerate(aovalues):
                        avalues[index] = worstcase_l1(outcomevalues,dist,bound)
                else:
                    for index,outcomevalues in enumerate(aovalues):
                        avalues[index] = -worstcase_l1(-outcomevalues,dist,bound)
                    
                action = avalues.argmax()
                outcome = outcomes[action]
                
                newvalues[stateindex] = avalues[action]
            
            policy[stateindex] = action
            scheme[stateindex] = outcome
            
        np.copyto(values, newvalues)
        residual = np.max(np.abs(newvalues - values))
        
    policy_mat = np.zeros((statecount,actioncount))
    for stateindex in range(statecount):
        policy_mat[stateindex,policy[stateindex]] = 1
        
    bound = (2 * residual / (1 - discount)) if discount < 1 else np.nan
    return values, policy_mat, bound
       
     
   


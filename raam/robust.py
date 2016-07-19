"""
================================================================
Plain python implementation of robust utilities (:mod:`raam.robust`)
================================================================

This implementation is slow, incomplete, and not as thoroughly
tested as the C++ version. Use at your own risk.
"""
import raam
import itertools
import numpy as np
from numpy import ma

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
  

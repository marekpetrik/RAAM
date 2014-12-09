"""
===============================================================
Simple benchmark chain problem (:mod:`raam.examples.chain`)
===============================================================

This probles does not use a simulator but instead constructs the samples directly.

See :mod:`simple_examples` for a description of the problem.
"""
import numpy as np
from math import floor
import raam


def simple_samples(length,succprob):
    """
    A simple example with actions for moving left and right with 
    succprob chance of the move and (1-succprob) chance of opposite move. 
    The end points are absorbing and the reward increases linearly from 
    the center. This is the direct representation of the samples, not the
    simulator.

    The representation of dec/exp states is table lookup.

    Parameters
    ----------
    length : int
        The number of states in the chain
    succprob : float
        The probability of the move in the desired direction.

    Returns
    -------
    out : raam.samples
        Samples
    """

    sc = length
    def ei(l,i):
        q = np.zeros(l)
        q[i] = 1
        return q
    #left and right actions
    def actions(self, decstate):
        return ['l','r']
    
    samples = raam.MemSamples()
        
    
    #left action
    for i in range(sc):
        samples.add_dec(raam.DecSample(ei(sc,i),'l',ei(2*sc,i),0,1))
    #right action
    for i in range(sc):
        samples.add_dec(raam.DecSample(ei(sc,i),'r',ei(2*sc,sc+i),0,1))

    #expstates are results of left-right actions
    expstates = []
    #left action
    for i in range(sc):
        nl = max(i-1,0)
        nr = min(i+1,sc-1)
        samples.add_exp(raam.ExpSample(ei(2*sc,i),ei(sc,nl),abs(nl - floor(sc/2)),succprob,1,0))
        samples.add_exp(raam.ExpSample(ei(2*sc,i),ei(sc,nr),abs(nr - floor(sc/2)),1-succprob,0,1,0))
        
    #right action
    for i in range(sc):
        nl = max(i-1,0)
        nr = min(i+1,sc-1)
        samples.add_exp(raam.ExpSample(ei(2*sc,sc+i),ei(sc,nl),abs(nl - floor(sc/2)),1-succprob,0,1,0))
        samples.add_exp(raam.ExpSample(ei(2*sc,sc+i),ei(sc,nr),abs(nr - floor(sc/2)),succprob,0,1,0))
    
    return samples

